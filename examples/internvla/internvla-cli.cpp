#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "../../vendor/stb/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static constexpr const char * DEFAULT_LLM_PATH = "/home/rock/ggml-hexagon/gguf/InternVLA-M1.Q4_K_M.gguf";
static constexpr const char * DEFAULT_VIT_PATH = "/home/rock/ggml-hexagon/gguf/InternVLA-M1.mmproj-f16.gguf";
static constexpr const char * DEFAULT_DIT_PATH = "/home/rock/ggml-hexagon/gguf/internvla_dit-f16.gguf";
static constexpr size_t INTERNVLA_BACKEND_SMOKE_MAX_WEIGHT_BYTES = 16 * 1024 * 1024;

struct internvla_args {
    std::string llm_path = DEFAULT_LLM_PATH;
    std::string vit_path = DEFAULT_VIT_PATH;
    std::string dit_path = DEFAULT_DIT_PATH;
    std::string dino_path;
    std::string qformer_path;
    std::string action_path;
    std::string backend = "Hexagon";
    std::vector<std::string> image_paths;
    std::string instruction;
    float cfg_scale = 1.5f;
    int ddim_steps = 10;
    uint32_t seed = 42;
    bool dump = false;
    bool dry_run = false;
    bool action_only = false;
    bool backend_smoke = true;
};

struct gguf_context_deleter {
    void operator()(gguf_context * ctx) const {
        if (ctx) {
            gguf_free(ctx);
        }
    }
};

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct ggml_backend_deleter {
    void operator()(ggml_backend * backend) const {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct ggml_backend_buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct llama_model_deleter {
    void operator()(llama_model * model) const {
        if (model) {
            llama_model_free(model);
        }
    }
};

struct llama_context_deleter {
    void operator()(llama_context * ctx) const {
        if (ctx) {
            llama_free(ctx);
        }
    }
};

struct llama_batch_deleter {
    void operator()(llama_batch * batch) const {
        if (batch) {
            llama_batch_free(*batch);
            delete batch;
        }
    }
};

template <typename T, typename D>
class unique_c_ptr {
public:
    unique_c_ptr() = default;
    explicit unique_c_ptr(T * ptr) : ptr(ptr) {}
    ~unique_c_ptr() { reset(); }

    unique_c_ptr(const unique_c_ptr &) = delete;
    unique_c_ptr & operator=(const unique_c_ptr &) = delete;

    unique_c_ptr(unique_c_ptr && other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    unique_c_ptr & operator=(unique_c_ptr && other) noexcept {
        if (this != &other) {
            reset();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    T * get() const { return ptr; }
    T ** out() { reset(); return &ptr; }
    explicit operator bool() const { return ptr != nullptr; }

    void reset(T * next = nullptr) {
        if (ptr) {
            D{}(ptr);
        }
        ptr = next;
    }

private:
    T * ptr = nullptr;
};

using gguf_ptr = unique_c_ptr<gguf_context, gguf_context_deleter>;
using ggml_ptr = unique_c_ptr<ggml_context, ggml_context_deleter>;
using ggml_backend_ptr = unique_c_ptr<ggml_backend, ggml_backend_deleter>;
using ggml_backend_buffer_ptr = unique_c_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter>;
using llama_model_ptr = unique_c_ptr<llama_model, llama_model_deleter>;
using llama_context_ptr = unique_c_ptr<llama_context, llama_context_deleter>;
using llama_batch_ptr = unique_c_ptr<llama_batch, llama_batch_deleter>;

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s [--llm LLM.gguf] [--vit VIT.gguf] [--dit INTERNVLA_DIT.gguf] [options]\n"
        "  %s --dry-run --dump\n\n"
        "Options:\n"
        "  --llm PATH          Qwen2VL language GGUF (default: %s)\n"
        "  --vit PATH          Qwen2VL mmproj/VIT GGUF (default: %s)\n"
        "  --dit PATH          InternVLA layer_qformer/action_model/dino bundle GGUF (default: %s)\n"
        "  --backend NAME      ggml backend for smoke graph execution (default: Hexagon)\n"
        "  --no-backend-smoke  Skip backend graph smoke execution\n"
        "  --dump              Print GGUF metadata and tensor summaries for all components\n"
        "  --dry-run           Load and validate GGUF files, then stop before inference\n"
        "  --action-only       Run the implemented ActionModel numeric projection graph only\n"
        "  --cfg-scale F       Classifier-free guidance scale (default: 1.5)\n"
        "  --ddim-steps N      DDIM sampling steps (default: 10)\n"
        "  --seed N            Random seed for initial diffusion noise (default: 42)\n"
        "  -h, --help          Show this help\n\n"
        "This tool is a standalone InternVLA-M1 runner. It loads the local Qwen2VL,\n"
        "Qwen2VL mmproj/VIT, and InternVLA DiT-bundle GGUF files, validates schemas,\n"
        "and runs backend smoke plus direct CPU DINO/QFormer/DiT/DDIM inference.\n",
        argv0, argv0, DEFAULT_LLM_PATH, DEFAULT_VIT_PATH, DEFAULT_DIT_PATH);
}

static bool parse_i32(const char * text, int & value) {
    char * end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    value = (int) parsed;
    return true;
}

static bool parse_u32(const char * text, uint32_t & value) {
    char * end = nullptr;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    value = (uint32_t) parsed;
    return true;
}

static bool parse_f32(const char * text, float & value) {
    char * end = nullptr;
    float parsed = std::strtof(text, &end);
    if (!end || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

static internvla_args parse_args(int argc, char ** argv) {
    internvla_args args;

    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (opt == "-h" || opt == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (opt == "--llm") {
            args.llm_path = need_value("--llm");
        } else if (opt == "--vit") {
            args.vit_path = need_value("--vit");
        } else if (opt == "--dit") {
            args.dit_path = need_value("--dit");
        } else if (opt == "--dino") {
            args.dino_path = need_value("--dino");
        } else if (opt == "--qformer") {
            args.qformer_path = need_value("--qformer");
        } else if (opt == "--action") {
            args.action_path = need_value("--action");
        } else if (opt == "--backend") {
            args.backend = need_value("--backend");
        } else if (opt == "--image") {
            args.image_paths.emplace_back(need_value("--image"));
        } else if (opt == "--instruction") {
            args.instruction = need_value("--instruction");
        } else if (opt == "--cfg-scale") {
            if (!parse_f32(need_value("--cfg-scale"), args.cfg_scale)) {
                throw std::runtime_error("invalid --cfg-scale");
            }
        } else if (opt == "--ddim-steps") {
            if (!parse_i32(need_value("--ddim-steps"), args.ddim_steps)) {
                throw std::runtime_error("invalid --ddim-steps");
            }
        } else if (opt == "--seed") {
            if (!parse_u32(need_value("--seed"), args.seed)) {
                throw std::runtime_error("invalid --seed");
            }
        } else if (opt == "--dump") {
            args.dump = true;
        } else if (opt == "--dry-run") {
            args.dry_run = true;
        } else if (opt == "--action-only") {
            args.action_only = true;
        } else if (opt == "--no-backend-smoke") {
            args.backend_smoke = false;
        } else {
            throw std::runtime_error("unknown argument: " + opt);
        }
    }

    if (args.dino_path.empty())    args.dino_path = args.dit_path;
    if (args.qformer_path.empty()) args.qformer_path = args.dit_path;
    if (args.action_path.empty())  args.action_path = args.dit_path;

    if (args.llm_path.empty())     throw std::runtime_error("missing --llm");
    if (args.vit_path.empty())     throw std::runtime_error("missing --vit");
    if (args.dit_path.empty())     throw std::runtime_error("missing --dit");
    if (args.backend.empty())      throw std::runtime_error("missing --backend");
    if (args.ddim_steps <= 0)      throw std::runtime_error("--ddim-steps must be positive");

    return args;
}

static std::string backend_devices_to_string() {
    std::string result;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!result.empty()) {
            result += ", ";
        }
        result += ggml_backend_dev_name(dev);
    }
    return result.empty() ? "<none>" : result;
}

static ggml_backend * init_backend_by_alias(const std::string & backend_name, std::string & resolved_name) {
    resolved_name = backend_name;
    ggml_backend * backend = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
    if (backend) {
        return backend;
    }

    if (backend_name == "Hexagon" || backend_name == "hexagon") {
        resolved_name = "HTP0";
        return ggml_backend_init_by_name(resolved_name.c_str(), nullptr);
    }

    return nullptr;
}

static bool read_file_at(const std::string & path, size_t offset, void * data, size_t size, std::string & error) {
    std::FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) {
        error = "failed to open " + path + ": " + std::strerror(errno);
        return false;
    }

    if (std::fseek(file, (long) offset, SEEK_SET) != 0) {
        error = "failed to seek " + path + " to offset " + std::to_string(offset);
        std::fclose(file);
        return false;
    }

    const size_t nread = std::fread(data, 1, size, file);
    std::fclose(file);
    if (nread != size) {
        error = "failed to read " + std::to_string(size) + " bytes from " + path + "; got " + std::to_string(nread);
        return false;
    }

    return true;
}

static ggml_tensor * build_linear(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    if (bias) {
        out = ggml_add(ctx, out, bias);
    }
    return out;
}

static float gelu_tanh(float x) {
    static constexpr float k_sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(k_sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

static float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void layer_norm_no_affine(std::vector<float> & x, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float mean = 0.0f;
        const int base = r * cols;
        for (int c = 0; c < cols; ++c) {
            mean += x[base + c];
        }
        mean /= (float) cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float d = x[base + c] - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt(var / (float) cols + eps);
        for (int c = 0; c < cols; ++c) {
            x[base + c] = (x[base + c] - mean) * inv;
        }
    }
}

static void layer_norm_affine(std::vector<float> & x, int rows, int cols, const std::vector<float> & weight, const std::vector<float> & bias, float eps) {
    for (int r = 0; r < rows; ++r) {
        float mean = 0.0f;
        const int base = r * cols;
        for (int c = 0; c < cols; ++c) {
            mean += x[base + c];
        }
        mean /= (float) cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float d = x[base + c] - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt(var / (float) cols + eps);
        for (int c = 0; c < cols; ++c) {
            x[base + c] = (x[base + c] - mean) * inv * weight[(size_t) c] + bias[(size_t) c];
        }
    }
}

static void rms_norm_affine(std::vector<float> & x, int rows, int cols, const std::vector<float> & weight, float eps) {
    for (int r = 0; r < rows; ++r) {
        const int base = r * cols;
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float v = x[base + c];
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / (float) cols + eps);
        for (int c = 0; c < cols; ++c) {
            x[base + c] = x[base + c] * inv * weight[(size_t) c];
        }
    }
}

static void linear_row_major(const std::vector<float> & input, int rows, int in_dim,
        const std::vector<float> & weight, const std::vector<float> & bias, int out_dim,
        std::vector<float> & output) {
    output.assign((size_t) rows * out_dim, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = bias.empty() ? 0.0f : bias[(size_t) o];
            for (int i = 0; i < in_dim; ++i) {
                sum += input[(size_t) r * in_dim + i] * weight[(size_t) o * in_dim + i];
            }
            output[(size_t) r * out_dim + o] = sum;
        }
    }
}

class internvla_op_runner {
public:
    explicit internvla_op_runner(const std::string & backend_name) {
        if (backend_name.empty() || backend_name == "CPU") {
            return;
        }
        backend.reset(init_backend_by_alias(backend_name, resolved_backend));
        if (!backend) {
            resolved_backend.clear();
        }
    }

    bool linear(const std::vector<float> & input, int rows, int in_dim,
            const std::vector<float> & weight, const std::vector<float> & bias, int out_dim,
            std::vector<float> & output) const {
        if (backend && linear_backend(input, rows, in_dim, weight, bias, out_dim, output)) {
            ++n_backend_linear;
            return true;
        }
        ++n_cpu_linear;
        linear_row_major(input, rows, in_dim, weight, bias, out_dim, output);
        return false;
    }

    bool layer_norm_no_affine(std::vector<float> & x, int rows, int cols, float eps) const {
        if (backend && norm_backend(x, rows, cols, nullptr, nullptr, eps)) {
            ++n_backend_norm;
            return true;
        }
        ++n_cpu_norm;
        ::layer_norm_no_affine(x, rows, cols, eps);
        return false;
    }

    bool layer_norm_affine(std::vector<float> & x, int rows, int cols,
            const std::vector<float> & weight, const std::vector<float> & bias, float eps) const {
        if (backend && norm_backend(x, rows, cols, &weight, &bias, eps)) {
            ++n_backend_norm;
            return true;
        }
        ++n_cpu_norm;
        ::layer_norm_affine(x, rows, cols, weight, bias, eps);
        return false;
    }

    bool silu_inplace(std::vector<float> & x) const {
        if (backend && unary_backend(x, GGML_UNARY_OP_SILU)) {
            ++n_backend_activation;
            return true;
        }
        ++n_cpu_activation;
        for (float & v : x) {
            v = silu(v);
        }
        return false;
    }

    bool gelu_tanh_inplace(std::vector<float> & x) const {
        if (backend && unary_backend(x, GGML_UNARY_OP_GELU)) {
            ++n_backend_activation;
            return true;
        }
        ++n_cpu_activation;
        for (float & v : x) {
            v = gelu_tanh(v);
        }
        return false;
    }

    void gelu_erf_cpu_inplace(std::vector<float> & x) const {
        ++n_cpu_activation;
        for (float & v : x) {
            v = gelu_erf(v);
        }
    }

    const char * backend_name() const {
        return backend ? ggml_backend_name(backend.get()) : "CPU";
    }

    void print_stats() const {
        std::printf("\nInternVLA op dispatch:\n");
        std::printf("  backend:        %s\n", backend_name());
        std::printf("  linear backend: %zu\n", n_backend_linear);
        std::printf("  linear CPU:     %zu\n", n_cpu_linear);
        std::printf("  norm backend:   %zu\n", n_backend_norm);
        std::printf("  norm CPU:       %zu\n", n_cpu_norm);
        std::printf("  act backend:    %zu\n", n_backend_activation);
        std::printf("  act CPU:        %zu\n", n_cpu_activation);
    }

private:
    bool linear_backend(const std::vector<float> & input, int rows, int in_dim,
            const std::vector<float> & weight, const std::vector<float> & bias, int out_dim,
            std::vector<float> & output) const {
        if (rows <= 0 || in_dim <= 0 || out_dim <= 0 ||
            input.size() != (size_t) rows * in_dim ||
            weight.size() != (size_t) out_dim * in_dim ||
            (!bias.empty() && bias.size() != (size_t) out_dim)) {
            return false;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 8 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_ptr ctx(ggml_init(params));
        if (!ctx) {
            return false;
        }

        ggml_tensor * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, in_dim, out_dim);
        ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, in_dim, rows);
        ggml_set_input(x);
        ggml_tensor * y = ggml_mul_mat(ctx.get(), w, x);
        ggml_tensor * b = nullptr;
        if (!bias.empty()) {
            b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, out_dim);
            y = ggml_add(ctx.get(), y, b);
        }
        ggml_set_output(y);

        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, y);
        if (!ggml_backend_supports_op(backend.get(), y)) {
            return false;
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            return false;
        }
        ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        if (b) {
            ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
        }

        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        output.resize((size_t) rows * out_dim);
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        return true;
    }

    bool norm_backend(std::vector<float> & x, int rows, int cols, const std::vector<float> * weight,
            const std::vector<float> * bias, float eps) const {
        if (rows <= 0 || cols <= 0 || x.size() != (size_t) rows * cols ||
            (weight && weight->size() != (size_t) cols) ||
            (bias && bias->size() != (size_t) cols)) {
            return false;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 8 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_ptr ctx(ggml_init(params));
        if (!ctx) {
            return false;
        }

        ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
        ggml_set_input(input);
        ggml_tensor * out = ggml_norm(ctx.get(), input, eps);
        ggml_tensor * w = nullptr;
        ggml_tensor * b = nullptr;
        if (weight) {
            w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, cols);
            out = ggml_mul(ctx.get(), out, w);
        }
        if (bias) {
            b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, cols);
            out = ggml_add(ctx.get(), out, b);
        }
        ggml_set_output(out);

        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, out);
        if (!ggml_backend_supports_op(backend.get(), out)) {
            return false;
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            return false;
        }
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        if (w) {
            ggml_backend_tensor_set(w, weight->data(), 0, weight->size() * sizeof(float));
        }
        if (b) {
            ggml_backend_tensor_set(b, bias->data(), 0, bias->size() * sizeof(float));
        }

        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(out, x.data(), 0, x.size() * sizeof(float));
        return true;
    }

    bool unary_backend(std::vector<float> & x, enum ggml_unary_op op) const {
        if (x.empty()) {
            return true;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 4 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_ptr ctx(ggml_init(params));
        if (!ctx) {
            return false;
        }

        ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t) x.size());
        ggml_set_input(input);
        ggml_tensor * out = ggml_unary(ctx.get(), input, op);
        ggml_set_output(out);

        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, out);
        if (!ggml_backend_supports_op(backend.get(), out)) {
            return false;
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            return false;
        }
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(out, x.data(), 0, x.size() * sizeof(float));
        return true;
    }

    std::string resolved_backend;
    ggml_backend_ptr backend;
    mutable size_t n_backend_linear = 0;
    mutable size_t n_cpu_linear = 0;
    mutable size_t n_backend_norm = 0;
    mutable size_t n_cpu_norm = 0;
    mutable size_t n_backend_activation = 0;
    mutable size_t n_cpu_activation = 0;
};

static void add_inplace(std::vector<float> & dst, const std::vector<float> & src) {
    GGML_ASSERT(dst.size() == src.size());
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

static void swiglu_inplace(std::vector<float> & gate, const std::vector<float> & up) {
    GGML_ASSERT(gate.size() == up.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = silu(gate[i]) * up[i];
    }
}

struct internvla_rgb_image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

static bool load_rgb_image(const std::string & path, internvla_rgb_image & image, std::string & error) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char * data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        error = "failed to decode image " + path + ": " + stbi_failure_reason();
        return false;
    }
    (void) channels;
    image.width = width;
    image.height = height;
    image.rgb.assign(data, data + (size_t) width * height * 3);
    stbi_image_free(data);
    return true;
}

static std::vector<float> resize_normalize_rgb(const internvla_rgb_image & image, int target, const float mean[3], const float stdv[3]) {
    std::vector<float> out((size_t) 3 * target * target);
    for (int y = 0; y < target; ++y) {
        const float src_y = ((float) y + 0.5f) * (float) image.height / (float) target - 0.5f;
        const int y0 = std::max(0, std::min(image.height - 1, (int) std::floor(src_y)));
        const int y1 = std::max(0, std::min(image.height - 1, y0 + 1));
        const float wy = src_y - (float) std::floor(src_y);
        for (int x = 0; x < target; ++x) {
            const float src_x = ((float) x + 0.5f) * (float) image.width / (float) target - 0.5f;
            const int x0 = std::max(0, std::min(image.width - 1, (int) std::floor(src_x)));
            const int x1 = std::max(0, std::min(image.width - 1, x0 + 1));
            const float wx = src_x - (float) std::floor(src_x);
            for (int c = 0; c < 3; ++c) {
                const float p00 = (float) image.rgb[((size_t) y0 * image.width + x0) * 3 + c];
                const float p01 = (float) image.rgb[((size_t) y0 * image.width + x1) * 3 + c];
                const float p10 = (float) image.rgb[((size_t) y1 * image.width + x0) * 3 + c];
                const float p11 = (float) image.rgb[((size_t) y1 * image.width + x1) * 3 + c];
                const float p0 = p00 * (1.0f - wx) + p01 * wx;
                const float p1 = p10 * (1.0f - wx) + p11 * wx;
                const float value = (p0 * (1.0f - wy) + p1 * wy) / 255.0f;
                out[(size_t) c * target * target + (size_t) y * target + x] = (value - mean[c]) / stdv[c];
            }
        }
    }
    return out;
}

static std::vector<float> resize_normalize_rgb224(const internvla_rgb_image & image) {
    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float stdv[3] = {0.229f, 0.224f, 0.225f};
    return resize_normalize_rgb(image, 224, mean, stdv);
}

static std::vector<float> resize_normalize_rgb560(const internvla_rgb_image & image, const std::vector<float> & mean, const std::vector<float> & stdv) {
    GGML_ASSERT(mean.size() == 3 && stdv.size() == 3);
    const float mean_arr[3] = { mean[0], mean[1], mean[2] };
    const float std_arr[3] = { stdv[0], stdv[1], stdv[2] };
    return resize_normalize_rgb(image, 560, mean_arr, std_arr);
}

static std::vector<float> timestep_embedding(int timestep, int dim) {
    std::vector<float> emb((size_t) dim, 0.0f);
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(10000.0f) * (float) i / (float) half);
        const float arg = (float) timestep * freq;
        emb[(size_t) i] = std::cos(arg);
        emb[(size_t) half + i] = std::sin(arg);
    }
    return emb;
}

static std::vector<double> betas_for_squaredcos(int n_steps) {
    std::vector<double> betas((size_t) n_steps);
    auto alpha_bar = [](double t) {
        const double v = std::cos((t + 0.008) / 1.008 * M_PI / 2.0);
        return v * v;
    };
    for (int i = 0; i < n_steps; ++i) {
        const double t1 = (double) i / (double) n_steps;
        const double t2 = (double) (i + 1) / (double) n_steps;
        betas[(size_t) i] = std::min(1.0 - alpha_bar(t2) / alpha_bar(t1), 0.999);
    }
    return betas;
}

static std::vector<int> ddim_timesteps(int diffusion_steps, int ddim_steps) {
    if (ddim_steps == 1) {
        return {50};
    }
    for (int stride = 1; stride < diffusion_steps; ++stride) {
        std::vector<int> steps;
        for (int t = 0; t < diffusion_steps; t += stride) {
            steps.push_back(t);
        }
        if ((int) steps.size() == ddim_steps) {
            return steps;
        }
    }
    std::vector<int> fallback;
    const double step = (double) (diffusion_steps - 1) / (double) std::max(1, ddim_steps - 1);
    for (int i = 0; i < ddim_steps; ++i) {
        fallback.push_back((int) std::llround(i * step));
    }
    return fallback;
}

static std::string kv_to_string(const gguf_context * ctx, int index) {
    const gguf_type type = gguf_get_kv_type(ctx, index);
    char buf[128];

    switch (type) {
        case GGUF_TYPE_UINT8:   std::snprintf(buf, sizeof(buf), "%u", (unsigned) gguf_get_val_u8(ctx, index)); return buf;
        case GGUF_TYPE_INT8:    std::snprintf(buf, sizeof(buf), "%d", (int) gguf_get_val_i8(ctx, index)); return buf;
        case GGUF_TYPE_UINT16:  std::snprintf(buf, sizeof(buf), "%u", (unsigned) gguf_get_val_u16(ctx, index)); return buf;
        case GGUF_TYPE_INT16:   std::snprintf(buf, sizeof(buf), "%d", (int) gguf_get_val_i16(ctx, index)); return buf;
        case GGUF_TYPE_UINT32:  std::snprintf(buf, sizeof(buf), "%u", gguf_get_val_u32(ctx, index)); return buf;
        case GGUF_TYPE_INT32:   std::snprintf(buf, sizeof(buf), "%d", gguf_get_val_i32(ctx, index)); return buf;
        case GGUF_TYPE_FLOAT32: std::snprintf(buf, sizeof(buf), "%.8g", (double) gguf_get_val_f32(ctx, index)); return buf;
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, index) ? "true" : "false";
        case GGUF_TYPE_STRING:  return gguf_get_val_str(ctx, index);
        case GGUF_TYPE_ARRAY: {
            const gguf_type arr_type = gguf_get_arr_type(ctx, index);
            const size_t n = gguf_get_arr_n(ctx, index);
            std::snprintf(buf, sizeof(buf), "array<%s>[%zu]", gguf_type_name(arr_type), n);
            return buf;
        }
        default:
            std::snprintf(buf, sizeof(buf), "<%s>", gguf_type_name(type));
            return buf;
    }
}

struct internvla_gguf_model {
    std::string label;
    std::string path;
    gguf_ptr ctx;
    ggml_ptr ctx_data;
    std::map<std::string, int> tensor_index;
    std::map<std::string, int> kv_index;

    bool load(const std::string & component_label, const std::string & component_path, bool read_data) {
        label = component_label;
        path = component_path;

        ggml_context * raw_ctx_data = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ !read_data,
            /*.ctx      =*/ &raw_ctx_data,
        };

        ctx.reset(gguf_init_from_file(path.c_str(), params));
        if (!ctx) {
            std::fprintf(stderr, "%s: failed to load %s GGUF: %s\n", __func__, label.c_str(), path.c_str());
            return false;
        }
        ctx_data.reset(raw_ctx_data);

        tensor_index.clear();
        for (int i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
            tensor_index.emplace(gguf_get_tensor_name(ctx.get(), i), i);
        }

        kv_index.clear();
        for (int i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
            kv_index.emplace(gguf_get_key(ctx.get(), i), i);
        }

        return true;
    }

    bool has_kv(const std::string & key) const {
        return kv_index.find(key) != kv_index.end();
    }

    bool has_tensor(const std::string & name) const {
        return tensor_index.find(name) != tensor_index.end();
    }

    bool has_tensor_prefix(const std::string & prefix) const {
        for (const auto & item : tensor_index) {
            if (item.first.rfind(prefix, 0) == 0) {
                return true;
            }
        }
        return false;
    }

    const ggml_tensor * tensor_meta(const std::string & name) const {
        if (!ctx_data) {
            return nullptr;
        }
        return ggml_get_tensor(ctx_data.get(), name.c_str());
    }

    bool read_tensor_bytes(const std::string & name, std::vector<uint8_t> & data, std::string & error) const {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end()) {
            error = label + " GGUF is missing tensor " + name;
            return false;
        }

        const int tensor_id = it->second;
        const size_t nbytes = gguf_get_tensor_size(ctx.get(), tensor_id);
        const size_t offset = gguf_get_data_offset(ctx.get()) + gguf_get_tensor_offset(ctx.get(), tensor_id);
        data.resize(nbytes);
        return read_file_at(path, offset, data.data(), data.size(), error);
    }

    bool read_tensor_f32(const std::string & name, std::vector<float> & data, std::vector<int64_t> * shape, std::string & error) const {
        const ggml_tensor * meta = tensor_meta(name);
        if (!meta) {
            error = label + " GGUF is missing tensor metadata " + name;
            return false;
        }

        std::vector<uint8_t> raw;
        if (!read_tensor_bytes(name, raw, error)) {
            return false;
        }

        const int64_t nelements = ggml_nelements(meta);
        data.resize((size_t) nelements);
        if (meta->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw.data()), data.data(), nelements);
        } else if (meta->type == GGML_TYPE_F32) {
            std::memcpy(data.data(), raw.data(), data.size() * sizeof(float));
        } else {
            error = name + " cannot be read as f32 from type " + ggml_type_name(meta->type);
            return false;
        }

        if (shape) {
            shape->assign(meta->ne, meta->ne + ggml_n_dims(meta));
        }
        return true;
    }

    std::string get_string(const std::string & key, const std::string & fallback = "") const {
        auto it = kv_index.find(key);
        if (it == kv_index.end()) {
            return fallback;
        }
        if (gguf_get_kv_type(ctx.get(), it->second) != GGUF_TYPE_STRING) {
            return fallback;
        }
        return gguf_get_val_str(ctx.get(), it->second);
    }

    uint32_t get_u32(const std::string & key, uint32_t fallback = 0) const {
        auto it = kv_index.find(key);
        if (it == kv_index.end()) {
            return fallback;
        }
        if (gguf_get_kv_type(ctx.get(), it->second) != GGUF_TYPE_UINT32) {
            return fallback;
        }
        return gguf_get_val_u32(ctx.get(), it->second);
    }

    float get_f32(const std::string & key, float fallback = 0.0f) const {
        auto it = kv_index.find(key);
        if (it == kv_index.end()) {
            return fallback;
        }
        if (gguf_get_kv_type(ctx.get(), it->second) != GGUF_TYPE_FLOAT32) {
            return fallback;
        }
        return gguf_get_val_f32(ctx.get(), it->second);
    }

    bool get_f32_array(const std::string & key, std::vector<float> & values) const {
        auto it = kv_index.find(key);
        if (it == kv_index.end()) {
            return false;
        }
        if (gguf_get_kv_type(ctx.get(), it->second) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx.get(), it->second) != GGUF_TYPE_FLOAT32) {
            return false;
        }
        const size_t n = gguf_get_arr_n(ctx.get(), it->second);
        const float * data = static_cast<const float *>(gguf_get_arr_data(ctx.get(), it->second));
        values.assign(data, data + n);
        return true;
    }

    void print_summary(bool verbose) const {
        std::printf("\n[%s]\n", label.c_str());
        std::printf("  path:       %s\n", path.c_str());
        std::printf("  version:    %d\n", gguf_get_version(ctx.get()));
        std::printf("  alignment:  %zu\n", gguf_get_alignment(ctx.get()));
        std::printf("  data_offs:  %zu\n", gguf_get_data_offset(ctx.get()));
        std::printf("  n_kv:       %lld\n", (long long) gguf_get_n_kv(ctx.get()));
        std::printf("  n_tensors:  %lld\n", (long long) gguf_get_n_tensors(ctx.get()));

        const std::string component = get_string("internvla.component");
        if (!component.empty()) {
            std::printf("  component:  %s\n", component.c_str());
        }
        const std::string arch = get_string("general.architecture");
        if (!arch.empty()) {
            std::printf("  arch:       %s\n", arch.c_str());
        }

        if (!verbose) {
            return;
        }

        std::printf("  metadata:\n");
        for (int i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
            const char * key = gguf_get_key(ctx.get(), i);
            std::string value = kv_to_string(ctx.get(), i);
            if (value.size() > 96) {
                value = value.substr(0, 93) + "...";
            }
            std::printf("    %-48s %s = %s\n", key, gguf_type_name(gguf_get_kv_type(ctx.get(), i)), value.c_str());
        }

        std::printf("  tensors:\n");
        for (int i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx.get(), i);
            const ggml_type type = gguf_get_tensor_type(ctx.get(), i);
            const size_t size = gguf_get_tensor_size(ctx.get(), i);
            const size_t offset = gguf_get_tensor_offset(ctx.get(), i);
            const ggml_tensor * meta = tensor_meta(name);
            if (meta) {
                std::printf("    %-64s %-8s shape=[%lld,%lld,%lld,%lld] size=%zu offset=%zu\n",
                        name, ggml_type_name(type),
                        (long long) meta->ne[0], (long long) meta->ne[1],
                        (long long) meta->ne[2], (long long) meta->ne[3], size, offset);
            } else {
                std::printf("    %-64s %-8s size=%zu offset=%zu\n", name, ggml_type_name(type), size, offset);
            }
        }
    }
};

class internvla_tensor_store {
public:
    explicit internvla_tensor_store(const internvla_gguf_model & model) : model(model) {}

    ggml_tensor * tensor_1d(ggml_context * ctx, const std::string & name, int64_t ne0, std::string & error) {
        return materialize(ctx, name, {ne0}, error);
    }

    ggml_tensor * tensor_1d_f32(ggml_context * ctx, const std::string & name, int64_t ne0, std::string & error) {
        return materialize_f32(ctx, name, {ne0}, error);
    }

    ggml_tensor * tensor_2d(ggml_context * ctx, const std::string & name, int64_t ne0, int64_t ne1, std::string & error) {
        return materialize(ctx, name, {ne0, ne1}, error);
    }

    void upload(ggml_tensor * tensor) const {
        const auto it = pending.find(tensor);
        if (it != pending.end()) {
            ggml_backend_tensor_set(tensor, it->second->data(), 0, it->second->size());
        }
    }

private:
    ggml_tensor * materialize(ggml_context * ctx, const std::string & name, const std::vector<int64_t> & expected, std::string & error) {
        const ggml_tensor * meta = validate_meta(name, expected, error);
        if (!meta) {
            return nullptr;
        }

        std::unique_ptr<std::vector<uint8_t>> bytes(new std::vector<uint8_t>());
        if (!model.read_tensor_bytes(name, *bytes, error)) {
            return nullptr;
        }

        ggml_tensor * tensor = ggml_new_tensor(ctx, meta->type, (int) expected.size(), expected.data());
        ggml_set_name(tensor, name.c_str());
        pending.emplace(tensor, std::move(bytes));
        return tensor;
    }

    ggml_tensor * materialize_f32(ggml_context * ctx, const std::string & name, const std::vector<int64_t> & expected, std::string & error) {
        const ggml_tensor * meta = validate_meta(name, expected, error);
        if (!meta) {
            return nullptr;
        }

        std::vector<uint8_t> raw;
        if (!model.read_tensor_bytes(name, raw, error)) {
            return nullptr;
        }

        const int64_t nelements = ggml_nelements(meta);
        std::unique_ptr<std::vector<uint8_t>> bytes(new std::vector<uint8_t>((size_t) nelements * sizeof(float)));
        float * dst = reinterpret_cast<float *>(bytes->data());
        if (meta->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw.data()), dst, nelements);
        } else if (meta->type == GGML_TYPE_F32) {
            std::memcpy(dst, raw.data(), bytes->size());
        } else {
            error = name + " cannot be materialized as f32 from type " + ggml_type_name(meta->type);
            return nullptr;
        }

        ggml_tensor * tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) expected.size(), expected.data());
        ggml_set_name(tensor, name.c_str());
        pending.emplace(tensor, std::move(bytes));
        return tensor;
    }

    const ggml_tensor * validate_meta(const std::string & name, const std::vector<int64_t> & expected, std::string & error) const {
        const ggml_tensor * meta = model.tensor_meta(name);
        if (!meta) {
            error = model.label + " GGUF is missing tensor metadata " + name;
            return nullptr;
        }
        if ((int) expected.size() != ggml_n_dims(meta)) {
            error = name + " has unexpected rank";
            return nullptr;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (meta->ne[i] != expected[i]) {
                error = name + " has unexpected shape";
                return nullptr;
            }
        }
        return meta;
    }

    const internvla_gguf_model & model;
    std::map<ggml_tensor *, std::unique_ptr<std::vector<uint8_t>>> pending;
};

struct internvla_backend_candidate {
    int tensor_id = -1;
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
    size_t nbytes = 0;
};

static bool is_backend_smoke_candidate(const ggml_tensor * tensor, size_t nbytes) {
    if (!tensor) {
        return false;
    }
    if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_n_dims(tensor) != 2 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        return false;
    }
    return nbytes <= INTERNVLA_BACKEND_SMOKE_MAX_WEIGHT_BYTES;
}

static bool find_backend_smoke_candidate(const internvla_gguf_model & model, internvla_backend_candidate & candidate) {
    static const char * preferred_prefixes[] = {
        "action_model.",
        "layer_qformer.",
        "dino_pro.",
        "dino_encoder.",
    };

    auto consider_prefix = [&](const char * prefix) {
        for (int i = 0; i < gguf_get_n_tensors(model.ctx.get()); ++i) {
            const char * name = gguf_get_tensor_name(model.ctx.get(), i);
            if (std::strncmp(name, prefix, std::strlen(prefix)) != 0) {
                continue;
            }
            const ggml_tensor * tensor = model.tensor_meta(name);
            const size_t nbytes = gguf_get_tensor_size(model.ctx.get(), i);
            if (!is_backend_smoke_candidate(tensor, nbytes)) {
                continue;
            }
            if (candidate.tensor_id >= 0 && nbytes >= candidate.nbytes) {
                continue;
            }
            candidate.tensor_id = i;
            candidate.name = name;
            candidate.type = tensor->type;
            candidate.nbytes = nbytes;
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                candidate.ne[dim] = tensor->ne[dim];
            }
        }
    };

    for (const char * prefix : preferred_prefixes) {
        consider_prefix(prefix);
        if (candidate.tensor_id >= 0) {
            return true;
        }
    }
    return false;
}

static bool run_backend_smoke_graph(const internvla_gguf_model & model, const std::string & backend_name, std::string & error) {
    internvla_backend_candidate candidate;
    if (!find_backend_smoke_candidate(model, candidate)) {
        error = "DiT bundle has no small 2D F16/F32 tensor suitable for backend smoke execution";
        return false;
    }

    std::string resolved_backend_name;
    ggml_backend_ptr backend(init_backend_by_alias(backend_name, resolved_backend_name));
    if (!backend) {
        error = "backend '" + backend_name + "' is not available. Available devices: " + backend_devices_to_string();
        return false;
    }

    std::vector<uint8_t> weight_data(candidate.nbytes);
    const size_t file_offset = gguf_get_data_offset(model.ctx.get()) + gguf_get_tensor_offset(model.ctx.get(), candidate.tensor_id);
    if (!read_file_at(model.path, file_offset, weight_data.data(), weight_data.size(), error)) {
        return false;
    }

    const int64_t input_cols = candidate.ne[0];
    const int64_t output_rows = candidate.ne[1];
    std::vector<float> input_data((size_t) input_cols);
    std::vector<float> output_data((size_t) output_rows);
    for (int64_t i = 0; i < input_cols; ++i) {
        input_data[(size_t) i] = (float) ((i % 7) + 1) * 0.01f;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_ptr compute_ctx(ggml_init(params));
    if (!compute_ctx) {
        error = "failed to create backend smoke ggml context";
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(compute_ctx.get(), candidate.type, candidate.ne[0], candidate.ne[1]);
    ggml_set_name(weight, candidate.name.c_str());
    ggml_tensor * input = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, candidate.ne[0], 1);
    ggml_set_name(input, "internvla.backend_smoke.input");
    ggml_set_input(input);
    ggml_tensor * output = ggml_mul_mat(compute_ctx.get(), weight, input);
    ggml_set_name(output, "internvla.backend_smoke.output");
    ggml_set_output(output);

    if (!ggml_backend_supports_op(backend.get(), output)) {
        error = "backend '" + backend_name + "' does not support the smoke mul_mat op for tensor " + candidate.name;
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph(compute_ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend.get()));
    if (!buffer) {
        error = "failed to allocate backend tensors on " + backend_name;
        return false;
    }

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "backend graph compute failed on " + backend_name + ": " + ggml_status_to_string(status);
        return false;
    }

    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));
    std::printf("\nInternVLA backend smoke run passed.\n");
    std::printf("  backend: %s\n", ggml_backend_name(backend.get()));
    std::printf("  tensor:  %s %s shape=[%lld,%lld] bytes=%zu\n",
            candidate.name.c_str(), ggml_type_name(candidate.type),
            (long long) candidate.ne[0], (long long) candidate.ne[1], candidate.nbytes);
    std::printf("  output:  ");
    const size_t n_print = std::min<size_t>(8, output_data.size());
    for (size_t i = 0; i < n_print; ++i) {
        std::printf("%.6f ", (double) output_data[i]);
    }
    std::printf("\n");
    return true;
}

struct internvla_pipeline_outputs {
    std::vector<float> qwen_hidden_states;
    std::vector<float> dino_encoded_features;
    std::vector<float> action_condition;
    std::vector<float> normalized_actions;
};

class internvla_qwen2vl_runner {
public:
    bool init(const internvla_gguf_model * llm_model, const internvla_gguf_model * vit_model) {
        llm = llm_model;
        vit = vit_model;
        return llm && vit;
    }

    bool validate_schema(std::string & error) const {
        if (!llm || !vit) {
            error = "Qwen2VL runner is not initialized";
            return false;
        }
        if (llm->get_string("general.architecture") != "qwen2vl") {
            error = "LLM GGUF general.architecture is not qwen2vl; got '" + llm->get_string("general.architecture") + "'";
            return false;
        }
        if (!llm->has_tensor("token_embd.weight")) {
            error = "LLM GGUF is missing required tensor token_embd.weight";
            return false;
        }
        if (!llm->has_tensor("output_norm.weight")) {
            error = "LLM GGUF is missing required tensor output_norm.weight";
            return false;
        }
        if (!vit->has_kv("clip.projector_type") && !vit->has_kv("clip.vision.projector_type")) {
            error = "ViT/mmproj GGUF is missing clip.projector_type or clip.vision.projector_type";
            return false;
        }
        return true;
    }

    bool forward_hidden_states(const std::vector<std::string> & image_paths, const std::string & instruction, internvla_pipeline_outputs & outputs, std::string & error) const {
        if (!llm) {
            error = "Qwen2VL runner is not initialized";
            return false;
        }
        std::vector<float> image_embeddings;
        qwen2vl_image_grid image_grid;
        if (!forward_image_embeddings(image_paths, image_embeddings, image_grid, error)) {
            return false;
        }

        llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap = true;
        llama_model_ptr model(llama_model_load_from_file(llm->path.c_str(), mparams));
        if (!model) {
            error = "failed to load Qwen2VL llama model from " + llm->path;
            return false;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 2048;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.n_seq_max = 1;
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cparams.n_threads = 4;
        cparams.n_threads_batch = 4;
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        if (!lctx) {
            error = "failed to create Qwen2VL llama context";
            return false;
        }
        llama_set_embeddings(lctx.get(), true);

        llama_pos n_past = 0;
        if (!image_embeddings.empty()) {
            if (!decode_qwen2vl_image_embeddings(lctx.get(), model.get(), image_embeddings, image_grid, 0, cparams.n_batch, n_past, error)) {
                return false;
            }
        }

        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::string prompt = instruction;
        if (prompt.empty()) {
            prompt = "predict robot action";
        }
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

        int n_tokens = -llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), nullptr, 0, true, true);
        if (n_tokens <= 0) {
            error = "failed to tokenize Qwen2VL prompt";
            return false;
        }
        std::vector<llama_token> tokens((size_t) n_tokens);
        const int check = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), tokens.data(), n_tokens, true, true);
        if (check < 0 || check != n_tokens) {
            error = "Qwen2VL prompt tokenization size mismatch";
            return false;
        }

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        llama_batch_ptr batch_guard(new llama_batch(batch));
        batch.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[(size_t) i];
            batch.pos[i] = n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        if (llama_decode(lctx.get(), batch) != 0) {
            error = "Qwen2VL llama_decode failed";
            return false;
        }

        const int n_embd = llama_model_n_embd(model.get());
        if (n_embd != 2048) {
            error = "Qwen2VL hidden size is " + std::to_string(n_embd) + ", expected 2048 for Layer-QFormer";
            return false;
        }

        const int keep_tokens = std::min(n_tokens, 64);
        outputs.qwen_hidden_states.resize((size_t) keep_tokens * n_embd);
        for (int i = 0; i < keep_tokens; ++i) {
            const int token_index = n_tokens - keep_tokens + i;
            float * embd = llama_get_embeddings_ith(lctx.get(), token_index);
            if (!embd) {
                error = "failed to read Qwen2VL embedding for token " + std::to_string(token_index);
                return false;
            }
            std::copy(embd, embd + n_embd, outputs.qwen_hidden_states.begin() + (size_t) i * n_embd);
        }
        return true;
    }

private:
    struct qwen2vl_image_grid {
        int nx = 0;
        int ny = 0;
    };

    struct qwen2vl_vision_hparams {
        uint32_t image_size = 0;
        uint32_t patch_size = 0;
        uint32_t hidden = 0;
        uint32_t ffn_hidden = 0;
        uint32_t projection = 0;
        uint32_t n_layer = 0;
        uint32_t n_head = 0;
        uint32_t window_pattern = 0;
        float eps = 1e-6f;
        std::vector<float> mean;
        std::vector<float> stdv;
    };

    struct qwen2vl_window_inputs {
        std::vector<int> idx;
        std::vector<int> inv_idx;
        std::vector<float> mask;
        std::vector<int> positions;
    };

    bool load_vision_hparams(qwen2vl_vision_hparams & hp, std::string & error) const {
        hp.image_size = vit->get_u32("clip.vision.image_size", 0);
        hp.patch_size = vit->get_u32("clip.vision.patch_size", 0);
        hp.hidden = vit->get_u32("clip.vision.embedding_length", 0);
        hp.ffn_hidden = vit->get_u32("clip.vision.feed_forward_length", 0);
        hp.projection = vit->get_u32("clip.vision.projection_dim", 0);
        hp.n_layer = vit->get_u32("clip.vision.block_count", 0);
        hp.n_head = vit->get_u32("clip.vision.attention.head_count", 0);
        hp.window_pattern = vit->get_u32("clip.vision.n_wa_pattern", 0);
        hp.eps = vit->get_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);

        const std::string projector = vit->get_string("clip.projector_type");
        if (hp.image_size != 560 || hp.patch_size != 14 || hp.hidden != 1280 || hp.ffn_hidden != 3420 ||
            hp.projection != 2048 || hp.n_layer != 32 || hp.n_head != 16 || hp.window_pattern != 8 ||
            projector != "qwen2.5vl_merger") {
            error = "unsupported Qwen2VL mmproj hparams for direct image path";
            return false;
        }
        if (!vit->get_f32_array("clip.vision.image_mean", hp.mean) || !vit->get_f32_array("clip.vision.image_std", hp.stdv) ||
            hp.mean.size() != 3 || hp.stdv.size() != 3) {
            error = "Qwen2VL mmproj GGUF is missing clip.vision.image_mean/std metadata";
            return false;
        }
        return true;
    }

    bool read_vit_vec(const std::string & name, std::vector<float> & data, std::string & error) const {
        return vit->read_tensor_f32(name, data, nullptr, error);
    }

    bool qwen2vl_patch_embed_tokens(const std::vector<float> & image_chw, const qwen2vl_vision_hparams & hp,
            std::vector<float> & patch_tokens, std::string & error) const {
        const int image_size = (int) hp.image_size;
        const int patch_size = (int) hp.patch_size;
        const int grid = image_size / patch_size;
        const int hidden = (int) hp.hidden;
        if (grid != 40 || hidden != 1280) {
            error = "Qwen2VL patch embed expects 560 image, 14 patch, 1280 hidden";
            return false;
        }

        std::vector<float> patch_w0, patch_w1;
        if (!read_vit_vec("v.patch_embd.weight", patch_w0, error) ||
            !read_vit_vec("v.patch_embd.weight.1", patch_w1, error)) {
            return false;
        }

        patch_tokens.assign((size_t) grid * grid * hidden, 0.0f);
        for (int py = 0; py < grid; ++py) {
            for (int px = 0; px < grid; ++px) {
                const int token = py * grid + px;
                for (int o = 0; o < hidden; ++o) {
                    float sum = 0.0f;
                    for (int c = 0; c < 3; ++c) {
                        for (int ky = 0; ky < patch_size; ++ky) {
                            const int iy = py * patch_size + ky;
                            for (int kx = 0; kx < patch_size; ++kx) {
                                const int ix = px * patch_size + kx;
                                const float v = image_chw[(size_t) c * image_size * image_size + (size_t) iy * image_size + ix];
                                const size_t w_idx = (((size_t) o * 3 + c) * patch_size + ky) * patch_size + kx;
                                sum += v * (patch_w0[w_idx] + patch_w1[w_idx]);
                            }
                        }
                    }
                    patch_tokens[(size_t) token * hidden + o] = sum;
                }
            }
        }
        return true;
    }

    static qwen2vl_window_inputs build_qwen2vl_window_inputs(const qwen2vl_vision_hparams & hp) {
        const int merge_ratio = 2;
        const int ipw = (int) hp.image_size / (int) hp.patch_size;
        const int iph = ipw;
        const int pw = ipw / merge_ratio;
        const int ph = iph / merge_ratio;
        const int n_patches = ipw * iph;
        const int mpow = merge_ratio * merge_ratio;

        qwen2vl_window_inputs inputs;
        inputs.idx.resize((size_t) ph * pw);
        inputs.inv_idx.resize((size_t) ph * pw);
        inputs.mask.assign((size_t) n_patches * n_patches, -INFINITY);

        const int attn_window_size = 112;
        const int grid_window = attn_window_size / (int) hp.patch_size / merge_ratio;
        int dst = 0;
        int mask_row = 0;
        for (int y = 0; y < ph; y += grid_window) {
            for (int x = 0; x < pw; x += grid_window) {
                const int win_h = std::min(grid_window, ph - y);
                const int win_w = std::min(grid_window, pw - x);
                const int dst_0 = dst;
                for (int dy = 0; dy < win_h; ++dy) {
                    for (int dx = 0; dx < win_w; ++dx) {
                        const int src = (y + dy) * pw + (x + dx);
                        inputs.idx[(size_t) src] = dst;
                        inputs.inv_idx[(size_t) dst] = src;
                        ++dst;
                    }
                }
                for (int r = 0; r < win_h * win_w * mpow; ++r) {
                    const int row_offset = mask_row * n_patches;
                    std::fill(inputs.mask.begin() + row_offset + dst_0 * mpow,
                              inputs.mask.begin() + row_offset + dst * mpow,
                              0.0f);
                    ++mask_row;
                }
            }
        }

        inputs.positions.resize((size_t) n_patches * 4);
        int ptr = 0;
        for (int y = 0; y < iph; y += merge_ratio) {
            for (int x = 0; x < ipw; x += merge_ratio) {
                for (int dy = 0; dy < merge_ratio; ++dy) {
                    for (int dx = 0; dx < merge_ratio; ++dx) {
                        int remap = inputs.idx[(size_t) ptr / mpow];
                        remap = remap * mpow + (ptr % mpow);
                        inputs.positions[(size_t) remap] = y + dy;
                        inputs.positions[(size_t) n_patches + remap] = x + dx;
                        inputs.positions[(size_t) 2 * n_patches + remap] = y + dy;
                        inputs.positions[(size_t) 3 * n_patches + remap] = x + dx;
                        ++ptr;
                    }
                }
            }
        }
        return inputs;
    }

    static void apply_qwen2vl_vision_rope(std::vector<float> & x, int n_tokens, int hidden, int n_head,
            const std::vector<int> & positions) {
        const int head_dim = hidden / n_head;
        const int rope_dim = head_dim / 2;
        const int quarter = rope_dim / 4;
        GGML_ASSERT(head_dim == 80 && rope_dim == 40 && quarter == 10);
        for (int token = 0; token < n_tokens; ++token) {
            const int pos_parts[4] = {
                positions[(size_t) token],
                positions[(size_t) n_tokens + token],
                positions[(size_t) 2 * n_tokens + token],
                positions[(size_t) 3 * n_tokens + token],
            };
            for (int h = 0; h < n_head; ++h) {
                const int base = token * hidden + h * head_dim;
                for (int section = 0; section < 4; ++section) {
                    const int section_base = section * quarter;
                    const int pos = pos_parts[section];
                    for (int i = 0; i < quarter; i += 2) {
                        const int d0 = section_base + i;
                        const int d1 = section_base + i + 1;
                        const float inv_freq = std::pow(10000.0f, -(float) i / (float) rope_dim);
                        const float angle = (float) pos * inv_freq;
                        const float cs = std::cos(angle);
                        const float sn = std::sin(angle);
                        const float a = x[(size_t) base + d0];
                        const float b = x[(size_t) base + d1];
                        x[(size_t) base + d0] = a * cs - b * sn;
                        x[(size_t) base + d1] = a * sn + b * cs;
                    }
                }
            }
        }
    }

    static void qwen2vl_attention(const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
            int n_tokens, int hidden, int n_head, const std::vector<float> * mask, std::vector<float> & attn) {
        const int head_dim = hidden / n_head;
        attn.assign((size_t) n_tokens * hidden, 0.0f);
        std::vector<float> scores((size_t) n_tokens);
        for (int h = 0; h < n_head; ++h) {
            const int base = h * head_dim;
            for (int tq = 0; tq < n_tokens; ++tq) {
                float max_score = -INFINITY;
                for (int tk = 0; tk < n_tokens; ++tk) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q[(size_t) tq * hidden + base + d] * k[(size_t) tk * hidden + base + d];
                    }
                    score /= std::sqrt((float) head_dim);
                    if (mask) {
                        score += (*mask)[(size_t) tq * n_tokens + tk];
                    }
                    scores[(size_t) tk] = score;
                    max_score = std::max(max_score, score);
                }
                float denom = 0.0f;
                for (int tk = 0; tk < n_tokens; ++tk) {
                    scores[(size_t) tk] = std::exp(scores[(size_t) tk] - max_score);
                    denom += scores[(size_t) tk];
                }
                for (int tk = 0; tk < n_tokens; ++tk) {
                    const float prob = scores[(size_t) tk] / denom;
                    for (int d = 0; d < head_dim; ++d) {
                        attn[(size_t) tq * hidden + base + d] += prob * v[(size_t) tk * hidden + base + d];
                    }
                }
            }
        }
    }

    bool qwen2vl_vision_block(int block, std::vector<float> & tokens, int n_tokens,
            const qwen2vl_vision_hparams & hp, const qwen2vl_window_inputs & window_inputs, bool full_attn,
            std::string & error) const {
        const int hidden = (int) hp.hidden;
        const int n_head = (int) hp.n_head;
        const int ffn_hidden = (int) hp.ffn_hidden;
        const std::string prefix = "v.blk." + std::to_string(block) + ".";

        std::vector<float> ln1_w, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
        std::vector<float> ln2_w, up_w, up_b, gate_w, gate_b, down_w, down_b;
        if (!read_vit_vec(prefix + "ln1.weight", ln1_w, error) ||
            !read_vit_vec(prefix + "attn_q.weight", q_w, error) ||
            !read_vit_vec(prefix + "attn_q.bias", q_b, error) ||
            !read_vit_vec(prefix + "attn_k.weight", k_w, error) ||
            !read_vit_vec(prefix + "attn_k.bias", k_b, error) ||
            !read_vit_vec(prefix + "attn_v.weight", v_w, error) ||
            !read_vit_vec(prefix + "attn_v.bias", v_b, error) ||
            !read_vit_vec(prefix + "attn_out.weight", o_w, error) ||
            !read_vit_vec(prefix + "attn_out.bias", o_b, error) ||
            !read_vit_vec(prefix + "ln2.weight", ln2_w, error) ||
            !read_vit_vec(prefix + "ffn_up.weight", up_w, error) ||
            !read_vit_vec(prefix + "ffn_up.bias", up_b, error) ||
            !read_vit_vec(prefix + "ffn_gate.weight", gate_w, error) ||
            !read_vit_vec(prefix + "ffn_gate.bias", gate_b, error) ||
            !read_vit_vec(prefix + "ffn_down.weight", down_w, error) ||
            !read_vit_vec(prefix + "ffn_down.bias", down_b, error)) {
            return false;
        }

        std::vector<float> x = tokens;
        rms_norm_affine(x, n_tokens, hidden, ln1_w, hp.eps);

        std::vector<float> q, k, v;
        linear_row_major(x, n_tokens, hidden, q_w, q_b, hidden, q);
        linear_row_major(x, n_tokens, hidden, k_w, k_b, hidden, k);
        linear_row_major(x, n_tokens, hidden, v_w, v_b, hidden, v);

        apply_qwen2vl_vision_rope(q, n_tokens, hidden, n_head, window_inputs.positions);
        apply_qwen2vl_vision_rope(k, n_tokens, hidden, n_head, window_inputs.positions);

        std::vector<float> attn;
        qwen2vl_attention(q, k, v, n_tokens, hidden, n_head, full_attn ? nullptr : &window_inputs.mask, attn);

        std::vector<float> attn_out;
        linear_row_major(attn, n_tokens, hidden, o_w, o_b, hidden, attn_out);
        add_inplace(tokens, attn_out);

        x = tokens;
        rms_norm_affine(x, n_tokens, hidden, ln2_w, hp.eps);
        std::vector<float> up, gate;
        linear_row_major(x, n_tokens, hidden, up_w, up_b, ffn_hidden, up);
        linear_row_major(x, n_tokens, hidden, gate_w, gate_b, ffn_hidden, gate);
        swiglu_inplace(gate, up);
        std::vector<float> ffn_out;
        linear_row_major(gate, n_tokens, ffn_hidden, down_w, down_b, hidden, ffn_out);
        add_inplace(tokens, ffn_out);
        return true;
    }

    bool qwen2vl_merger(const std::vector<float> & vision_tokens, const qwen2vl_vision_hparams & hp,
            const qwen2vl_window_inputs & window_inputs, std::vector<float> & embeddings, std::string & error) const {
        const int grid = (int) hp.image_size / (int) hp.patch_size;
        const int merged_grid = grid / 2;
        const int hidden = (int) hp.hidden;
        const int merged_dim = hidden * 4;
        const int merged_tokens = merged_grid * merged_grid;
        if (vision_tokens.size() != (size_t) grid * grid * hidden) {
            error = "Qwen2VL merger received unexpected vision token shape";
            return false;
        }

        std::vector<float> post_w;
        std::vector<float> mm0_w, mm0_b, mm2_w, mm2_b;
        if (!read_vit_vec("v.post_ln.weight", post_w, error) ||
            !read_vit_vec("mm.0.weight", mm0_w, error) ||
            !read_vit_vec("mm.0.bias", mm0_b, error) ||
            !read_vit_vec("mm.2.weight", mm2_w, error) ||
            !read_vit_vec("mm.2.bias", mm2_b, error)) {
            return false;
        }

        std::vector<float> normed = vision_tokens;
        rms_norm_affine(normed, grid * grid, hidden, post_w, hp.eps);

        std::vector<float> merged((size_t) merged_tokens * merged_dim, 0.0f);
        for (int my = 0; my < merged_grid; ++my) {
            for (int mx = 0; mx < merged_grid; ++mx) {
                const int out_token = my * merged_grid + mx;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int src_token = (my * 2 + dy) * grid + (mx * 2 + dx);
                        const int quad = dy * 2 + dx;
                        std::copy(normed.begin() + (size_t) src_token * hidden,
                                  normed.begin() + (size_t) (src_token + 1) * hidden,
                                  merged.begin() + ((size_t) out_token * 4 + quad) * hidden);
                    }
                }
            }
        }

        std::vector<float> hidden_proj;
        linear_row_major(merged, merged_tokens, merged_dim, mm0_w, mm0_b, merged_dim, hidden_proj);
        for (float & v : hidden_proj) {
            v = gelu_erf(v);
        }
        std::vector<float> projected;
        linear_row_major(hidden_proj, merged_tokens, merged_dim, mm2_w, mm2_b, (int) hp.projection, projected);

        embeddings.assign((size_t) merged_tokens * hp.projection, 0.0f);
        for (int src = 0; src < merged_tokens; ++src) {
            const int dst = window_inputs.idx[(size_t) src];
            std::copy(projected.begin() + (size_t) src * hp.projection,
                      projected.begin() + (size_t) (src + 1) * hp.projection,
                      embeddings.begin() + (size_t) dst * hp.projection);
        }
        return true;
    }

    bool decode_qwen2vl_image_embeddings(llama_context * lctx, const llama_model * model,
            const std::vector<float> & embeddings, const qwen2vl_image_grid & grid, llama_seq_id seq_id,
            int32_t n_batch, llama_pos & n_past, std::string & error) const {
        const int n_embd_inp = llama_model_n_embd_inp(model);
        if (n_embd_inp <= 0 || (int) embeddings.size() % n_embd_inp != 0) {
            error = "Qwen2VL image embedding size is not divisible by llama input embedding dimension";
            return false;
        }
        const int n_tokens = (int) embeddings.size() / n_embd_inp;
        if (grid.nx <= 0 || grid.ny <= 0 || n_tokens != grid.nx * grid.ny) {
            error = "Qwen2VL image embedding count does not match image grid";
            return false;
        }
        if (n_embd_inp != 2048) {
            error = "Qwen2VL llama input embedding size is " + std::to_string(n_embd_inp) + ", expected 2048";
            return false;
        }
        if (n_batch <= 0) {
            error = "Qwen2VL llama context batch size is invalid";
            return false;
        }
        if (llama_model_rope_type(model) != LLAMA_ROPE_TYPE_MROPE) {
            error = "Qwen2VL image embedding decode requires an M-RoPE llama model";
            return false;
        }

        const int cur_batch_size = std::min(n_batch, n_tokens);
        std::vector<llama_pos> pos((size_t) cur_batch_size * 4);
        std::vector<int32_t> n_seq_id((size_t) cur_batch_size, 1);
        std::vector<llama_seq_id> seq_id_values((size_t) cur_batch_size, seq_id);
        std::vector<llama_seq_id *> seq_id_ptrs((size_t) cur_batch_size + 1, nullptr);
        std::vector<int8_t> logits((size_t) cur_batch_size, false);
        for (int i = 0; i < cur_batch_size; ++i) {
            seq_id_ptrs[(size_t) i] = &seq_id_values[(size_t) i];
        }

        llama_batch batch = {};
        batch.token = nullptr;
        batch.pos = pos.data();
        batch.n_seq_id = n_seq_id.data();
        batch.seq_id = seq_id_ptrs.data();
        batch.logits = logits.data();

        const llama_pos pos_0 = n_past;
        int offset = 0;
        while (offset < n_tokens) {
            const int cur = std::min(cur_batch_size, n_tokens - offset);
            batch.n_tokens = cur;
            batch.embd = const_cast<float *>(embeddings.data() + (size_t) offset * n_embd_inp);
            for (int i = 0; i < cur; ++i) {
                const int token_index = offset + i;
                pos[(size_t) i] = pos_0;
                pos[(size_t) cur + i] = pos_0 + token_index / grid.nx;
                pos[(size_t) 2 * cur + i] = pos_0 + token_index % grid.nx;
                pos[(size_t) 3 * cur + i] = 0;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq_id;
                batch.logits[i] = false;
            }
            if (llama_decode(lctx, batch) != 0) {
                error = "Qwen2VL llama_decode failed while injecting image embeddings";
                return false;
            }
            offset += cur;
        }

        n_past += std::max(grid.nx, grid.ny);
        return true;
    }

    bool forward_image_embeddings(const std::vector<std::string> & image_paths, std::vector<float> & embeddings,
            qwen2vl_image_grid & grid, std::string & error) const {
        embeddings.clear();
        grid = {};
        if (image_paths.empty()) {
            return true;
        }
        if (!vit) {
            error = "Qwen2VL mmproj runner is not initialized";
            return false;
        }

        qwen2vl_vision_hparams hp;
        if (!load_vision_hparams(hp, error)) {
            return false;
        }
        static const char * required_tensors[] = {
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.blk.0.attn_q.weight",
            "v.blk.31.attn_out.weight",
            "v.post_ln.weight",
            "mm.0.weight",
            "mm.2.weight",
        };
        for (const char * name : required_tensors) {
            if (!vit->has_tensor(name)) {
                error = std::string("Qwen2VL mmproj GGUF is missing tensor ") + name;
                return false;
            }
        }

        std::vector<std::vector<float>> images_chw;
        images_chw.reserve(image_paths.size());
        for (const std::string & image_path : image_paths) {
            internvla_rgb_image image;
            if (!load_rgb_image(image_path, image, error)) {
                return false;
            }
            images_chw.emplace_back(resize_normalize_rgb560(image, hp.mean, hp.stdv));
        }

        std::vector<float> patch_tokens;
        if (!qwen2vl_patch_embed_tokens(images_chw.front(), hp, patch_tokens, error)) {
            return false;
        }
        if (patch_tokens.size() != (size_t) 40 * 40 * 1280) {
            error = "Qwen2VL patch embedding produced unexpected token shape";
            return false;
        }

        qwen2vl_window_inputs window_inputs = build_qwen2vl_window_inputs(hp);
        if (window_inputs.idx.size() != (size_t) 20 * 20 ||
            window_inputs.inv_idx.size() != (size_t) 20 * 20 ||
            window_inputs.mask.size() != (size_t) 40 * 40 * 40 * 40 ||
            window_inputs.positions.size() != (size_t) 40 * 40 * 4) {
            error = "Qwen2VL window attention inputs have unexpected shape";
            return false;
        }

        for (int block = 0; block < (int) hp.n_layer; ++block) {
            const bool full_attn = ((block + 1) % (int) hp.window_pattern) == 0;
            if (!qwen2vl_vision_block(block, patch_tokens, 40 * 40, hp, window_inputs, full_attn, error)) {
                return false;
            }
        }
        if (!qwen2vl_merger(patch_tokens, hp, window_inputs, embeddings, error)) {
            return false;
        }
        if (embeddings.size() != (size_t) 20 * 20 * 2048) {
            error = "Qwen2VL merger produced unexpected embedding shape";
            return false;
        }
        grid.nx = 20;
        grid.ny = 20;
        return true;
    }

    const internvla_gguf_model * llm = nullptr;
    const internvla_gguf_model * vit = nullptr;
};

class internvla_dino_runner {
public:
    bool init(const internvla_gguf_model * model) {
        dino = model;
        return dino != nullptr;
    }

    bool validate_schema(std::string & error) const {
        if (!dino) {
            error = "DINO runner is not initialized";
            return false;
        }
        const std::string component = dino->get_string("internvla.component");
        if (component == "dit_bundle") {
            if (!dino->has_tensor_prefix("dino_encoder.") || !dino->has_tensor_prefix("dino_pro.")) {
                error = "DiT bundle is missing dino_encoder.* or dino_pro.* tensors";
                return false;
            }
        } else if (!component.empty() && component != "dino_encoder" && component != "dino_encoder_dino_pro") {
            error = "DINO GGUF internvla.component should be dino_encoder, dino_encoder_dino_pro, or dit_bundle; got '" + component + "'";
            return false;
        }
        return true;
    }

        bool forward(const std::vector<std::string> & image_paths, const internvla_op_runner & ops,
            internvla_pipeline_outputs & outputs, std::string & error) const {
        if (!dino) {
            error = "DINO runner is not initialized";
            return false;
        }

        std::vector<std::vector<float>> images_chw;
        if (image_paths.empty()) {
            images_chw.emplace_back((size_t) 3 * 224 * 224, 0.0f);
        } else {
            images_chw.reserve(image_paths.size());
            for (const std::string & image_path : image_paths) {
                internvla_rgb_image image;
                if (!load_rgb_image(image_path, image, error)) {
                    return false;
                }
                images_chw.emplace_back(resize_normalize_rgb224(image));
            }
        }

        const int n_views = std::max<int>(1, (int) image_paths.size());
        const int n_patch_tokens = 256;
        const int dino_hidden = 384;
        const int projected_hidden = 2048;
        outputs.dino_encoded_features.clear();
        outputs.dino_encoded_features.reserve((size_t) n_views * n_patch_tokens * projected_hidden);

        std::vector<float> dino_pro_w, dino_pro_b;
        if (!read_vec("dino_pro.weight", dino_pro_w, error) ||
            !read_vec("dino_pro.bias", dino_pro_b, error)) {
            return false;
        }

        for (const std::vector<float> & image_chw : images_chw) {
            std::vector<float> patch_tokens;
            if (!forward_dinov2_s14(image_chw, ops, patch_tokens, error)) {
                return false;
            }
            std::vector<float> projected;
            ops.linear(patch_tokens, n_patch_tokens, dino_hidden, dino_pro_w, dino_pro_b, projected_hidden, projected);
            outputs.dino_encoded_features.insert(outputs.dino_encoded_features.end(), projected.begin(), projected.end());
        }
        return true;
    }

private:
    bool read_vec(const std::string & name, std::vector<float> & data, std::string & error) const {
        return dino->read_tensor_f32(name, data, nullptr, error);
    }

        bool forward_dinov2_s14(const std::vector<float> & image_chw, const internvla_op_runner & ops,
            std::vector<float> & patch_tokens, std::string & error) const {
        static constexpr int image_size = 224;
        static constexpr int patch = 14;
        static constexpr int grid = image_size / patch;
        static constexpr int n_patches = grid * grid;
        static constexpr int n_tokens = n_patches + 1;
        static constexpr int hidden = 384;
        static constexpr int heads = 6;
        static constexpr int head_dim = hidden / heads;
        static constexpr int mlp_hidden = 1536;

        std::vector<float> cls_token, pos_embed, patch_w, patch_b;
        if (!read_vec("dino_encoder.body.cls_token", cls_token, error) ||
            !read_vec("dino_encoder.body.pos_embed", pos_embed, error) ||
            !read_vec("dino_encoder.body.patch_embed.proj.weight", patch_w, error) ||
            !read_vec("dino_encoder.body.patch_embed.proj.bias", patch_b, error)) {
            return false;
        }

        std::vector<float> x((size_t) n_tokens * hidden, 0.0f);
        std::copy(cls_token.begin(), cls_token.begin() + hidden, x.begin());
        for (int py = 0; py < grid; ++py) {
            for (int px = 0; px < grid; ++px) {
                const int token = 1 + py * grid + px;
                for (int o = 0; o < hidden; ++o) {
                    float sum = patch_b[(size_t) o];
                    for (int c = 0; c < 3; ++c) {
                        for (int ky = 0; ky < patch; ++ky) {
                            const int iy = py * patch + ky;
                            for (int kx = 0; kx < patch; ++kx) {
                                const int ix = px * patch + kx;
                                const float v = image_chw[(size_t) c * image_size * image_size + (size_t) iy * image_size + ix];
                                const float w = patch_w[(((size_t) o * 3 + c) * patch + ky) * patch + kx];
                                sum += v * w;
                            }
                        }
                    }
                    x[(size_t) token * hidden + o] = sum;
                }
            }
        }

        add_dino_pos_embed(x, pos_embed, grid, hidden);

        for (int block = 0; block < 12; ++block) {
            if (!dino_block(block, x, n_tokens, hidden, heads, head_dim, mlp_hidden, ops, error)) {
                return false;
            }
        }

        std::vector<float> norm_w, norm_b;
        if (!read_vec("dino_encoder.body.norm.weight", norm_w, error) ||
            !read_vec("dino_encoder.body.norm.bias", norm_b, error)) {
            return false;
        }
        ops.layer_norm_affine(x, n_tokens, hidden, norm_w, norm_b, 1e-6f);

        patch_tokens.resize((size_t) n_patches * hidden);
        std::copy(x.begin() + hidden, x.end(), patch_tokens.begin());
        return true;
    }

    static void add_dino_pos_embed(std::vector<float> & x, const std::vector<float> & pos_embed, int out_grid, int hidden) {
        const int src_grid = 37;
        for (int d = 0; d < hidden; ++d) {
            x[(size_t) d] += pos_embed[(size_t) d];
        }
        for (int y = 0; y < out_grid; ++y) {
            const float src_y = ((float) y + 0.5f) * (float) src_grid / (float) out_grid - 0.5f;
            const int y0 = std::max(0, std::min(src_grid - 1, (int) std::floor(src_y)));
            const int y1 = std::max(0, std::min(src_grid - 1, y0 + 1));
            const float wy = src_y - (float) std::floor(src_y);
            for (int x_pos = 0; x_pos < out_grid; ++x_pos) {
                const float src_x = ((float) x_pos + 0.5f) * (float) src_grid / (float) out_grid - 0.5f;
                const int x0 = std::max(0, std::min(src_grid - 1, (int) std::floor(src_x)));
                const int x1 = std::max(0, std::min(src_grid - 1, x0 + 1));
                const float wx = src_x - (float) std::floor(src_x);
                const int dst_token = 1 + y * out_grid + x_pos;
                for (int d = 0; d < hidden; ++d) {
                    const float p00 = pos_embed[((size_t) 1 + y0 * src_grid + x0) * hidden + d];
                    const float p01 = pos_embed[((size_t) 1 + y0 * src_grid + x1) * hidden + d];
                    const float p10 = pos_embed[((size_t) 1 + y1 * src_grid + x0) * hidden + d];
                    const float p11 = pos_embed[((size_t) 1 + y1 * src_grid + x1) * hidden + d];
                    const float p0 = p00 * (1.0f - wx) + p01 * wx;
                    const float p1 = p10 * (1.0f - wx) + p11 * wx;
                    x[(size_t) dst_token * hidden + d] += p0 * (1.0f - wy) + p1 * wy;
                }
            }
        }
    }

        bool dino_block(int block, std::vector<float> & x, int n_tokens, int hidden, int heads, int head_dim,
            int mlp_hidden, const internvla_op_runner & ops, std::string & error) const {
        const std::string prefix = "dino_encoder.body.blocks." + std::to_string(block) + ".";
        std::vector<float> norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b, ls1;
        std::vector<float> norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b, ls2;
        if (!read_vec(prefix + "norm1.weight", norm1_w, error) ||
            !read_vec(prefix + "norm1.bias", norm1_b, error) ||
            !read_vec(prefix + "attn.qkv.weight", qkv_w, error) ||
            !read_vec(prefix + "attn.qkv.bias", qkv_b, error) ||
            !read_vec(prefix + "attn.proj.weight", proj_w, error) ||
            !read_vec(prefix + "attn.proj.bias", proj_b, error) ||
            !read_vec(prefix + "ls1.gamma", ls1, error) ||
            !read_vec(prefix + "norm2.weight", norm2_w, error) ||
            !read_vec(prefix + "norm2.bias", norm2_b, error) ||
            !read_vec(prefix + "mlp.fc1.weight", fc1_w, error) ||
            !read_vec(prefix + "mlp.fc1.bias", fc1_b, error) ||
            !read_vec(prefix + "mlp.fc2.weight", fc2_w, error) ||
            !read_vec(prefix + "mlp.fc2.bias", fc2_b, error) ||
            !read_vec(prefix + "ls2.gamma", ls2, error)) {
            return false;
        }

        std::vector<float> y = x;
        ops.layer_norm_affine(y, n_tokens, hidden, norm1_w, norm1_b, 1e-6f);

        std::vector<float> qkv;
        ops.linear(y, n_tokens, hidden, qkv_w, qkv_b, hidden * 3, qkv);
        std::vector<float> attn_out((size_t) n_tokens * hidden, 0.0f);
        std::vector<float> scores((size_t) n_tokens);
        for (int h = 0; h < heads; ++h) {
            const int head_base = h * head_dim;
            for (int tq = 0; tq < n_tokens; ++tq) {
                float max_score = -INFINITY;
                for (int tk = 0; tk < n_tokens; ++tk) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        const int dim = head_base + d;
                        const float q = qkv[(size_t) tq * hidden * 3 + dim];
                        const float k = qkv[(size_t) tk * hidden * 3 + hidden + dim];
                        score += q * k;
                    }
                    score /= std::sqrt((float) head_dim);
                    scores[(size_t) tk] = score;
                    max_score = std::max(max_score, score);
                }
                float denom = 0.0f;
                for (int tk = 0; tk < n_tokens; ++tk) {
                    scores[(size_t) tk] = std::exp(scores[(size_t) tk] - max_score);
                    denom += scores[(size_t) tk];
                }
                for (int tk = 0; tk < n_tokens; ++tk) {
                    const float prob = scores[(size_t) tk] / denom;
                    for (int d = 0; d < head_dim; ++d) {
                        const int dim = head_base + d;
                        const float v = qkv[(size_t) tk * hidden * 3 + hidden * 2 + dim];
                        attn_out[(size_t) tq * hidden + dim] += prob * v;
                    }
                }
            }
        }

        std::vector<float> projected;
        ops.linear(attn_out, n_tokens, hidden, proj_w, proj_b, hidden, projected);
        for (int t = 0; t < n_tokens; ++t) {
            for (int d = 0; d < hidden; ++d) {
                x[(size_t) t * hidden + d] += projected[(size_t) t * hidden + d] * ls1[(size_t) d];
            }
        }

        y = x;
        ops.layer_norm_affine(y, n_tokens, hidden, norm2_w, norm2_b, 1e-6f);
        std::vector<float> mlp;
        ops.linear(y, n_tokens, hidden, fc1_w, fc1_b, mlp_hidden, mlp);
        ops.gelu_erf_cpu_inplace(mlp);
        std::vector<float> mlp_out;
        ops.linear(mlp, n_tokens, mlp_hidden, fc2_w, fc2_b, hidden, mlp_out);
        for (int t = 0; t < n_tokens; ++t) {
            for (int d = 0; d < hidden; ++d) {
                x[(size_t) t * hidden + d] += mlp_out[(size_t) t * hidden + d] * ls2[(size_t) d];
            }
        }
        return true;
    }

    const internvla_gguf_model * dino = nullptr;
};

class internvla_qformer_runner {
public:
    bool init(const internvla_gguf_model * model) {
        qformer = model;
        return qformer != nullptr;
    }

    bool validate_schema(std::string & error) const {
        if (!qformer) {
            error = "Layer-QFormer runner is not initialized";
            return false;
        }
        const std::string component = qformer->get_string("internvla.component");
        if (component == "dit_bundle") {
            if (!qformer->has_tensor_prefix("layer_qformer.")) {
                error = "DiT bundle is missing layer_qformer.* tensors";
                return false;
            }
        } else if (!component.empty() && component != "layer_qformer") {
            error = "QFormer GGUF internvla.component should be layer_qformer or dit_bundle; got '" + component + "'";
            return false;
        }
        return true;
    }

    bool forward(internvla_pipeline_outputs & outputs, const internvla_op_runner & ops, std::string & error) const {
        if (!qformer) {
            error = "Layer-QFormer runner is not initialized";
            return false;
        }

        static constexpr int input_dim = 2048;
        static constexpr int hidden = 768;
        static constexpr int query_tokens = 64;
        static constexpr int n_heads = 8;
        static constexpr int head_dim = hidden / n_heads;
        static constexpr int mlp_hidden = hidden * 4;

        std::vector<float> encoder = outputs.qwen_hidden_states;
        if (!outputs.dino_encoded_features.empty()) {
            encoder.insert(encoder.end(), outputs.dino_encoded_features.begin(), outputs.dino_encoded_features.end());
        }
        if (encoder.empty()) {
            encoder.assign((size_t) query_tokens * input_dim, 0.0f);
        }
        if (encoder.size() % input_dim != 0) {
            error = "QFormer encoder input must have feature dim 2048";
            return false;
        }
        const int encoder_tokens = (int) encoder.size() / input_dim;

        std::vector<float> q_tokens, proj_w, proj_b, norm1_w, norm1_b, in_proj_w, in_proj_b, out_w, out_b;
        std::vector<float> norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b;
        if (!read_vec("layer_qformer.query_tokens", q_tokens, error) ||
            !read_vec("layer_qformer.proj.weight", proj_w, error) ||
            !read_vec("layer_qformer.proj.bias", proj_b, error) ||
            !read_vec("layer_qformer.layers.0.norm1.weight", norm1_w, error) ||
            !read_vec("layer_qformer.layers.0.norm1.bias", norm1_b, error) ||
            !read_vec("layer_qformer.layers.0.cross_attn.in_proj_weight", in_proj_w, error) ||
            !read_vec("layer_qformer.layers.0.cross_attn.in_proj_bias", in_proj_b, error) ||
            !read_vec("layer_qformer.layers.0.cross_attn.out_proj.weight", out_w, error) ||
            !read_vec("layer_qformer.layers.0.cross_attn.out_proj.bias", out_b, error) ||
            !read_vec("layer_qformer.layers.0.norm2.weight", norm2_w, error) ||
            !read_vec("layer_qformer.layers.0.norm2.bias", norm2_b, error) ||
            !read_vec("layer_qformer.layers.0.mlp.0.weight", fc1_w, error) ||
            !read_vec("layer_qformer.layers.0.mlp.0.bias", fc1_b, error) ||
            !read_vec("layer_qformer.layers.0.mlp.2.weight", fc2_w, error) ||
            !read_vec("layer_qformer.layers.0.mlp.2.bias", fc2_b, error)) {
            return false;
        }

        std::vector<float> kv;
        ops.linear(encoder, encoder_tokens, input_dim, proj_w, proj_b, hidden, kv);
        std::vector<float> query = q_tokens;
        ops.layer_norm_affine(query, query_tokens, hidden, norm1_w, norm1_b, 1e-5f);

        std::vector<float> q((size_t) query_tokens * hidden, 0.0f);
        std::vector<float> k((size_t) encoder_tokens * hidden, 0.0f);
        std::vector<float> v((size_t) encoder_tokens * hidden, 0.0f);
        std::vector<float> q_w(in_proj_w.begin(), in_proj_w.begin() + (size_t) hidden * hidden);
        std::vector<float> q_b(in_proj_b.begin(), in_proj_b.begin() + hidden);
        std::vector<float> k_w(in_proj_w.begin() + (size_t) hidden * hidden,
                               in_proj_w.begin() + (size_t) 2 * hidden * hidden);
        std::vector<float> k_b(in_proj_b.begin() + hidden, in_proj_b.begin() + 2 * hidden);
        std::vector<float> v_w(in_proj_w.begin() + (size_t) 2 * hidden * hidden,
                               in_proj_w.begin() + (size_t) 3 * hidden * hidden);
        std::vector<float> v_b(in_proj_b.begin() + 2 * hidden, in_proj_b.begin() + 3 * hidden);
        ops.linear(query, query_tokens, hidden, q_w, q_b, hidden, q);
        ops.linear(kv, encoder_tokens, hidden, k_w, k_b, hidden, k);
        ops.linear(kv, encoder_tokens, hidden, v_w, v_b, hidden, v);

        std::vector<float> attn((size_t) query_tokens * hidden, 0.0f);
        const float scale = 1.0f / std::sqrt((float) head_dim);
        for (int head = 0; head < n_heads; ++head) {
            for (int qi = 0; qi < query_tokens; ++qi) {
                std::vector<float> scores((size_t) encoder_tokens);
                float max_score = -INFINITY;
                for (int ki = 0; ki < encoder_tokens; ++ki) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        const int idx = head * head_dim + d;
                        dot += q[(size_t) qi * hidden + idx] * k[(size_t) ki * hidden + idx];
                    }
                    scores[(size_t) ki] = dot * scale;
                    max_score = std::max(max_score, scores[(size_t) ki]);
                }
                float sum_score = 0.0f;
                for (float & s : scores) {
                    s = std::exp(s - max_score);
                    sum_score += s;
                }
                for (int vi = 0; vi < encoder_tokens; ++vi) {
                    const float w = scores[(size_t) vi] / sum_score;
                    for (int d = 0; d < head_dim; ++d) {
                        const int idx = head * head_dim + d;
                        attn[(size_t) qi * hidden + idx] += w * v[(size_t) vi * hidden + idx];
                    }
                }
            }
        }

        std::vector<float> attn_proj;
    ops.linear(attn, query_tokens, hidden, out_w, out_b, hidden, attn_proj);
        query = q_tokens;
        add_inplace(query, attn_proj);
        std::vector<float> mlp_in = query;
        ops.layer_norm_affine(mlp_in, query_tokens, hidden, norm2_w, norm2_b, 1e-5f);
        std::vector<float> fc1;
    ops.linear(mlp_in, query_tokens, hidden, fc1_w, fc1_b, mlp_hidden, fc1);
        ops.gelu_tanh_inplace(fc1);
        std::vector<float> fc2;
        ops.linear(fc1, query_tokens, mlp_hidden, fc2_w, fc2_b, hidden, fc2);
        add_inplace(query, fc2);
        outputs.action_condition = std::move(query);
        return true;
    }

private:
    bool read_vec(const std::string & name, std::vector<float> & out, std::string & error) const {
        std::vector<int64_t> shape;
        if (!qformer->read_tensor_f32(name, out, &shape, error)) {
            return false;
        }
        return true;
    }

    const internvla_gguf_model * qformer = nullptr;
};

class internvla_action_runner {
public:
    bool init(const internvla_gguf_model * model) {
        action = model;
        return action != nullptr;
    }

    bool validate_schema(std::string & error) const {
        if (!action) {
            error = "Action runner is not initialized";
            return false;
        }
        const std::string component = action->get_string("internvla.component");
        if (component == "dit_bundle") {
            if (!action->has_tensor_prefix("action_model.")) {
                error = "DiT bundle is missing action_model.* tensors";
                return false;
            }
        } else if (!component.empty() && component != "action_model") {
            error = "Action GGUF internvla.component should be action_model or dit_bundle; got '" + component + "'";
            return false;
        }
        return true;
    }

    bool sample_ddim(const std::string & backend_name, const internvla_op_runner & ops,
            float cfg_scale, int ddim_steps, uint32_t seed, internvla_pipeline_outputs & outputs, std::string & error) const {
        if (sample_ddim_cpu(ops, cfg_scale, ddim_steps, seed, outputs, error)) {
            return true;
        }

        if (!action) {
            error = "Action runner is not initialized";
            return false;
        }

        std::string resolved_backend_name;
        ggml_backend_ptr backend(init_backend_by_alias(backend_name, resolved_backend_name));
        if (!backend) {
            error = "backend '" + backend_name + "' is not available for ActionModel numeric graph. Available devices: " + backend_devices_to_string();
            return false;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_ptr gctx(ggml_init(params));
        if (!gctx) {
            error = "failed to create ActionModel ggml context";
            return false;
        }

        internvla_tensor_store store(*action);
        ggml_tensor * hist_w = store.tensor_2d(gctx.get(), "action_model.net.history_embedder.linear.weight", 7, 768, error);
        ggml_tensor * hist_b = store.tensor_1d_f32(gctx.get(), "action_model.net.history_embedder.linear.bias", 768, error);
        ggml_tensor * x_w    = store.tensor_2d(gctx.get(), "action_model.net.x_embedder.linear.weight", 7, 768, error);
        ggml_tensor * x_b    = store.tensor_1d_f32(gctx.get(), "action_model.net.x_embedder.linear.bias", 768, error);
        ggml_tensor * out_w  = store.tensor_2d(gctx.get(), "action_model.net.final_layer.linear.weight", 768, 7, error);
        ggml_tensor * out_b  = store.tensor_1d_f32(gctx.get(), "action_model.net.final_layer.linear.bias", 7, error);
        if (!hist_w || !hist_b || !x_w || !x_b || !out_w || !out_b) {
            return false;
        }

        ggml_tensor * hist_in = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_F32, 7, 1);
        ggml_set_name(hist_in, "internvla.action.history_input");
        ggml_set_input(hist_in);
        ggml_tensor * x_in = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_F32, 7, 1);
        ggml_set_name(x_in, "internvla.action.noise_input");
        ggml_set_input(x_in);

        ggml_tensor * hist = build_linear(gctx.get(), hist_in, hist_w, hist_b);
        ggml_tensor * x    = build_linear(gctx.get(), x_in,    x_w,    x_b);
        ggml_tensor * hidden = ggml_add(gctx.get(), hist, x);
        ggml_tensor * pred = build_linear(gctx.get(), hidden, out_w, out_b);
        ggml_set_name(pred, "internvla.action.projection_output");
        ggml_set_output(pred);

        if (!ggml_backend_supports_op(backend.get(), pred)) {
            error = std::string("backend ") + ggml_backend_name(backend.get()) + " does not support ActionModel projection graph";
            return false;
        }

        ggml_cgraph * graph = ggml_new_graph(gctx.get());
        ggml_build_forward_expand(graph, pred);

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(gctx.get(), backend.get()));
        if (!buffer) {
            error = "failed to allocate ActionModel backend tensors";
            return false;
        }

        store.upload(hist_w);
        store.upload(hist_b);
        store.upload(x_w);
        store.upload(x_b);
        store.upload(out_w);
        store.upload(out_b);

        float hist_data[7] = {0};
        float noise_data[7] = {0};
        uint32_t s = seed ? seed : 1;
        for (int i = 0; i < 7; ++i) {
            s = 1664525u * s + 1013904223u;
            noise_data[i] = ((float) (s & 0xffff) / 32768.0f) - 1.0f;
        }
        ggml_backend_tensor_set(hist_in, hist_data, 0, sizeof(hist_data));
        ggml_backend_tensor_set(x_in, noise_data, 0, sizeof(noise_data));

        const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            error = std::string("ActionModel projection graph failed: ") + ggml_status_to_string(status);
            return false;
        }

        outputs.normalized_actions.resize(7);
        ggml_backend_tensor_get(pred, outputs.normalized_actions.data(), 0, outputs.normalized_actions.size() * sizeof(float));
        return true;
    }

private:
    bool read_vec(const std::string & name, std::vector<float> & out, std::string & error) const {
        std::vector<int64_t> shape;
        if (!action->read_tensor_f32(name, out, &shape, error)) {
            return false;
        }
        return true;
    }

        bool dit_forward_cpu(const std::vector<float> & x_in, int timestep, const std::vector<float> & z_in,
            const internvla_op_runner & ops, float cfg_scale, bool use_cfg, std::vector<float> & eps_out,
            std::string & error) const {
        static constexpr int action_dim = 7;
        static constexpr int hidden = 768;
        static constexpr int cond_tokens = 64;
        static constexpr int action_tokens = 16;
        static constexpr int total_tokens = cond_tokens + action_tokens;
        static constexpr int n_heads = 12;
        static constexpr int head_dim = hidden / n_heads;
        static constexpr int mlp_hidden = hidden * 4;

        const int batch = use_cfg ? 2 : 1;
        std::vector<float> x_embed_w, x_embed_b, t0_w, t0_b, t2_w, t2_b, z_w, z_b, pos, uncond;
        std::vector<float> final_w, final_b;
        if (!read_vec("action_model.net.x_embedder.linear.weight", x_embed_w, error) ||
            !read_vec("action_model.net.x_embedder.linear.bias",   x_embed_b, error) ||
            !read_vec("action_model.net.t_embedder.mlp.0.weight",  t0_w, error) ||
            !read_vec("action_model.net.t_embedder.mlp.0.bias",    t0_b, error) ||
            !read_vec("action_model.net.t_embedder.mlp.2.weight",  t2_w, error) ||
            !read_vec("action_model.net.t_embedder.mlp.2.bias",    t2_b, error) ||
            !read_vec("action_model.net.z_embedder.linear.weight", z_w, error) ||
            !read_vec("action_model.net.z_embedder.linear.bias",   z_b, error) ||
            !read_vec("action_model.net.z_embedder.uncondition",   uncond, error) ||
            !read_vec("action_model.net.positional_embedding",     pos, error) ||
            !read_vec("action_model.net.final_layer.linear.weight", final_w, error) ||
            !read_vec("action_model.net.final_layer.linear.bias",   final_b, error)) {
            return false;
        }

        std::vector<float> t_freq = timestep_embedding(timestep, 256);
        std::vector<float> t_hidden;
        ops.linear(t_freq, 1, 256, t0_w, t0_b, hidden, t_hidden);
        ops.silu_inplace(t_hidden);
        std::vector<float> t_emb;
        ops.linear(t_hidden, 1, hidden, t2_w, t2_b, hidden, t_emb);

        std::vector<float> tokens((size_t) batch * total_tokens * hidden, 0.0f);
        for (int b = 0; b < batch; ++b) {
            const bool unconditional = use_cfg && b == 1;
            const std::vector<float> & z_src = unconditional ? uncond : z_in;
            std::vector<float> z_proj;
            ops.linear(z_src, cond_tokens, hidden, z_w, z_b, hidden, z_proj);
            for (int q = 0; q < cond_tokens; ++q) {
                for (int h = 0; h < hidden; ++h) {
                    tokens[((size_t) b * total_tokens + q) * hidden + h] = z_proj[(size_t) q * hidden + h] + t_emb[(size_t) h];
                }
            }

            std::vector<float> x_proj;
            ops.linear(x_in, action_tokens, action_dim, x_embed_w, x_embed_b, hidden, x_proj);
            for (int t = 0; t < action_tokens; ++t) {
                for (int h = 0; h < hidden; ++h) {
                    tokens[((size_t) b * total_tokens + cond_tokens + t) * hidden + h] = x_proj[(size_t) t * hidden + h];
                }
            }

            for (int tok = 0; tok < total_tokens; ++tok) {
                for (int h = 0; h < hidden; ++h) {
                    tokens[((size_t) b * total_tokens + tok) * hidden + h] += pos[(size_t) tok * hidden + h];
                }
            }
        }

        for (int il = 0; il < 12; ++il) {
            const std::string prefix = "action_model.net.blocks." + std::to_string(il) + ".";
            std::vector<float> qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b;
            if (!read_vec(prefix + "attn.qkv.weight", qkv_w, error) ||
                !read_vec(prefix + "attn.qkv.bias",   qkv_b, error) ||
                !read_vec(prefix + "attn.proj.weight", proj_w, error) ||
                !read_vec(prefix + "attn.proj.bias",   proj_b, error) ||
                !read_vec(prefix + "mlp.fc1.weight", fc1_w, error) ||
                !read_vec(prefix + "mlp.fc1.bias",   fc1_b, error) ||
                !read_vec(prefix + "mlp.fc2.weight", fc2_w, error) ||
                !read_vec(prefix + "mlp.fc2.bias",   fc2_b, error)) {
                return false;
            }

            for (int b = 0; b < batch; ++b) {
                std::vector<float> block_in(tokens.begin() + (size_t) b * total_tokens * hidden,
                                            tokens.begin() + (size_t) (b + 1) * total_tokens * hidden);
                std::vector<float> norm = block_in;
                ops.layer_norm_no_affine(norm, total_tokens, hidden, 1e-6f);

                std::vector<float> qkv;
                ops.linear(norm, total_tokens, hidden, qkv_w, qkv_b, hidden * 3, qkv);
                std::vector<float> attn_out((size_t) total_tokens * hidden, 0.0f);
                const float scale = 1.0f / std::sqrt((float) head_dim);
                for (int head = 0; head < n_heads; ++head) {
                    for (int qi = 0; qi < total_tokens; ++qi) {
                        std::vector<float> scores((size_t) total_tokens);
                        float max_score = -INFINITY;
                        for (int ki = 0; ki < total_tokens; ++ki) {
                            float dot = 0.0f;
                            for (int d = 0; d < head_dim; ++d) {
                                const int idx = head * head_dim + d;
                                const float q = qkv[(size_t) qi * hidden * 3 + idx];
                                const float k = qkv[(size_t) ki * hidden * 3 + hidden + idx];
                                dot += q * k;
                            }
                            scores[(size_t) ki] = dot * scale;
                            max_score = std::max(max_score, scores[(size_t) ki]);
                        }
                        float sum_score = 0.0f;
                        for (float & s : scores) {
                            s = std::exp(s - max_score);
                            sum_score += s;
                        }
                        for (int vi = 0; vi < total_tokens; ++vi) {
                            const float w = scores[(size_t) vi] / sum_score;
                            for (int d = 0; d < head_dim; ++d) {
                                const int idx = head * head_dim + d;
                                const float v = qkv[(size_t) vi * hidden * 3 + hidden * 2 + idx];
                                attn_out[(size_t) qi * hidden + idx] += w * v;
                            }
                        }
                    }
                }
                std::vector<float> attn_proj;
                ops.linear(attn_out, total_tokens, hidden, proj_w, proj_b, hidden, attn_proj);
                add_inplace(block_in, attn_proj);

                std::vector<float> mlp_in = block_in;
                ops.layer_norm_no_affine(mlp_in, total_tokens, hidden, 1e-6f);
                std::vector<float> fc1;
                ops.linear(mlp_in, total_tokens, hidden, fc1_w, fc1_b, mlp_hidden, fc1);
                ops.gelu_tanh_inplace(fc1);
                std::vector<float> fc2;
                ops.linear(fc1, total_tokens, mlp_hidden, fc2_w, fc2_b, hidden, fc2);
                add_inplace(block_in, fc2);
                std::copy(block_in.begin(), block_in.end(), tokens.begin() + (size_t) b * total_tokens * hidden);
            }
        }

        std::vector<float> eps_batch((size_t) batch * action_tokens * 7, 0.0f);
        for (int b = 0; b < batch; ++b) {
            std::vector<float> action_tokens_hidden((size_t) action_tokens * hidden);
            for (int t = 0; t < action_tokens; ++t) {
                const size_t src = ((size_t) b * total_tokens + cond_tokens + t) * hidden;
                std::copy(tokens.begin() + src, tokens.begin() + src + hidden, action_tokens_hidden.begin() + (size_t) t * hidden);
            }
            ops.layer_norm_no_affine(action_tokens_hidden, action_tokens, hidden, 1e-6f);
            std::vector<float> pred;
            ops.linear(action_tokens_hidden, action_tokens, hidden, final_w, final_b, 7, pred);
            std::copy(pred.begin(), pred.end(), eps_batch.begin() + (size_t) b * action_tokens * 7);
        }

        eps_out.assign((size_t) action_tokens * 7, 0.0f);
        if (use_cfg) {
            for (size_t i = 0; i < eps_out.size(); ++i) {
                const float cond = eps_batch[i];
                const float uncond_eps = eps_batch[eps_out.size() + i];
                eps_out[i] = uncond_eps + cfg_scale * (cond - uncond_eps);
            }
        } else {
            std::copy(eps_batch.begin(), eps_batch.begin() + eps_out.size(), eps_out.begin());
        }
        return true;
    }

        bool sample_ddim_cpu(const internvla_op_runner & ops, float cfg_scale, int ddim_steps, uint32_t seed,
            internvla_pipeline_outputs & outputs, std::string & error) const {
        if (!action) {
            error = "Action runner is not initialized";
            return false;
        }
        static constexpr int action_tokens = 16;
        static constexpr int action_dim = 7;
        static constexpr int hidden = 768;
        static constexpr int cond_tokens = 64;
        const bool use_cfg = cfg_scale > 1.0f;

        std::vector<float> z = outputs.action_condition;
        if (z.empty()) {
            z.assign((size_t) cond_tokens * hidden, 0.0f);
        }
        if ((int) z.size() != cond_tokens * hidden) {
            error = "Action condition must have shape [64,768]";
            return false;
        }

        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> x((size_t) action_tokens * action_dim);
        for (float & v : x) {
            v = normal(rng);
        }

        std::vector<double> betas = betas_for_squaredcos(100);
        std::vector<double> base_alpha_cumprod(100);
        double prod = 1.0;
        for (int i = 0; i < 100; ++i) {
            prod *= 1.0 - betas[(size_t) i];
            base_alpha_cumprod[(size_t) i] = prod;
        }
        std::vector<int> steps = ddim_timesteps(100, ddim_steps);
        std::vector<double> alpha_cumprod(steps.size());
        double last_alpha = 1.0;
        double spaced_prod = 1.0;
        for (size_t i = 0; i < steps.size(); ++i) {
            const double alpha = base_alpha_cumprod[(size_t) steps[i]];
            const double new_beta = 1.0 - alpha / last_alpha;
            spaced_prod *= 1.0 - new_beta;
            alpha_cumprod[i] = spaced_prod;
            last_alpha = alpha;
        }

        for (int step_i = (int) steps.size() - 1; step_i >= 0; --step_i) {
            const int t_index = step_i;
            const int original_t = steps[(size_t) step_i];
            std::vector<float> eps;
            if (!dit_forward_cpu(x, original_t, z, ops, cfg_scale, use_cfg, eps, error)) {
                return false;
            }

            const double alpha_bar = alpha_cumprod[(size_t) t_index];
            const double alpha_bar_prev = t_index == 0 ? 1.0 : alpha_cumprod[(size_t) t_index - 1];
            const double sqrt_recip = std::sqrt(1.0 / alpha_bar);
            const double sqrt_recipm1 = std::sqrt(1.0 / alpha_bar - 1.0);
            const double sqrt_alpha_prev = std::sqrt(alpha_bar_prev);
            const double sqrt_one_minus_alpha_prev = std::sqrt(std::max(0.0, 1.0 - alpha_bar_prev));

            for (size_t i = 0; i < x.size(); ++i) {
                const double pred_x0 = sqrt_recip * x[i] - sqrt_recipm1 * eps[i];
                x[i] = (float) (sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * eps[i]);
            }
        }

        outputs.normalized_actions = x;
        return true;
    }

    const internvla_gguf_model * action = nullptr;
};

class internvla_context {
public:
    bool load(const internvla_args & args) {
        return llm.load("llm", args.llm_path, false)
            && vit.load("vit", args.vit_path, false)
            && dino.load("dino", args.dino_path, false)
            && qformer.load("layer_qformer", args.qformer_path, false)
            && action.load("action_model", args.action_path, false);
    }

    void dump(bool verbose) const {
        llm.print_summary(verbose);
        vit.print_summary(verbose);
        dino.print_summary(verbose);
        qformer.print_summary(verbose);
        action.print_summary(verbose);
    }

    bool validate(std::string & error) {
        qwen_runner.init(&llm, &vit);
        dino_runner.init(&dino);
        qformer_runner.init(&qformer);
        action_runner.init(&action);

        if (!qwen_runner.validate_schema(error)) return false;
        if (!dino_runner.validate_schema(error)) return false;
        if (!qformer_runner.validate_schema(error)) return false;
        if (!action_runner.validate_schema(error)) return false;
        return true;
    }

    bool predict_action(const internvla_args & args, internvla_pipeline_outputs & outputs, std::string & error) {
        internvla_op_runner ops(args.backend);
        if (args.image_paths.empty()) {
            error = "missing --image for full action prediction; use --dry-run for load/schema/backend validation only";
            return false;
        }
        if (args.instruction.empty()) {
            error = "missing --instruction for full action prediction; use --dry-run for load/schema/backend validation only";
            return false;
        }
        if (!qwen_runner.forward_hidden_states(args.image_paths, args.instruction, outputs, error)) {
            return false;
        }
        if (!dino_runner.forward(args.image_paths, ops, outputs, error)) {
            return false;
        }
        if (!qformer_runner.forward(outputs, ops, error)) {
            return false;
        }
        if (!action_runner.sample_ddim(args.backend, ops, args.cfg_scale, args.ddim_steps, args.seed, outputs, error)) {
            return false;
        }
        ops.print_stats();
        return true;
    }

    bool predict_action_only(const internvla_args & args, internvla_pipeline_outputs & outputs, std::string & error) {
        internvla_op_runner ops(args.backend);
        if (!action_runner.sample_ddim(args.backend, ops, args.cfg_scale, args.ddim_steps, args.seed, outputs, error)) {
            return false;
        }
        ops.print_stats();
        return true;
    }

    bool run_backend_smoke(const internvla_args & args, std::string & error) const {
        return run_backend_smoke_graph(action, args.backend, error);
    }

private:
    internvla_gguf_model llm;
    internvla_gguf_model vit;
    internvla_gguf_model dino;
    internvla_gguf_model qformer;
    internvla_gguf_model action;

    internvla_qwen2vl_runner qwen_runner;
    internvla_dino_runner dino_runner;
    internvla_qformer_runner qformer_runner;
    internvla_action_runner action_runner;
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_backend_load_all();

    internvla_args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 1;
    }

    internvla_context ctx;
    if (!ctx.load(args)) {
        return 1;
    }

    ctx.dump(args.dump);

    std::string error;
    if (!ctx.validate(error)) {
        std::fprintf(stderr, "schema validation failed: %s\n", error.c_str());
        return 2;
    }

    std::printf("\nInternVLA-M1 GGUF schema validation passed.\n");
    std::printf("  images:      %zu\n", args.image_paths.size());
    std::printf("  instruction: %s\n", args.instruction.empty() ? "<none>" : args.instruction.c_str());
    std::printf("  cfg_scale:   %.3f\n", (double) args.cfg_scale);
    std::printf("  ddim_steps:  %d\n", args.ddim_steps);
    std::printf("  seed:        %u\n", args.seed);
    std::printf("  backend:     %s\n", args.backend.c_str());

    if (args.backend_smoke && !ctx.run_backend_smoke(args, error)) {
        std::fprintf(stderr, "backend smoke run failed: %s\n", error.c_str());
        return 3;
    }

    if (args.action_only) {
        internvla_pipeline_outputs outputs;
        if (!ctx.predict_action_only(args, outputs, error)) {
            std::fprintf(stderr, "action-only inference failed: %s\n", error.c_str());
            return 4;
        }

        std::printf("{\"normalized_actions\":[");
        for (size_t i = 0; i < outputs.normalized_actions.size(); ++i) {
            if (i > 0) {
                std::printf(",");
            }
            std::printf("%.9g", (double) outputs.normalized_actions[i]);
        }
        std::printf("]}\n");
        return 0;
    }

    if (args.dry_run || args.dump) {
        std::printf("\nDry-run complete. Full numeric inference kernels are staged in component runners.\n");
        return 0;
    }

    internvla_pipeline_outputs outputs;
    if (!ctx.predict_action(args, outputs, error)) {
        std::fprintf(stderr, "inference failed: %s\n", error.c_str());
        return 4;
    }

    std::printf("{\"normalized_actions\":[");
    for (size_t i = 0; i < outputs.normalized_actions.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("%.9g", (double) outputs.normalized_actions[i]);
    }
    std::printf("]}\n");

    return 0;
}
