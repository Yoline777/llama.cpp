#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
cd "$repo_root"

preset=${PRESET:-arm64-android-snapdragon-release}
build_dir=${BUILD_DIR:-build-snapdragon}
pkg_dir=${PKG_DIR:-$build_dir/pkg-snapdragon/llama.cpp}
remote_root=${REMOTE_ROOT:-/data/local/tmp}
remote_pkg=${REMOTE_PKG:-$remote_root/llama.cpp}
remote_gguf=${REMOTE_GGUF:-$remote_root/gguf}
cli_bin=${CLI_BIN:-llama-internvla-cli}
backend=${BACKEND:-HTP0}

llm_model=${LLM_MODEL:-/home/rock/ggml-hexagon/gguf/InternVLA-M1.Q4_K_M.gguf}
vit_model=${VIT_MODEL:-/home/rock/ggml-hexagon/gguf/InternVLA-M1.mmproj-f16.gguf}
dit_model=${DIT_MODEL:-/home/rock/ggml-hexagon/gguf/internvla_dit-f16.gguf}

jobs=${JOBS:-$(nproc)}
run_args=${RUN_ARGS:---dry-run}

find_adb() {
    if [[ -n "${ADB:-}" ]]; then
        printf '%s\n' "$ADB"
        return 0
    fi
    if command -v adb >/dev/null 2>&1; then
        command -v adb
        return 0
    fi
    for candidate in \
        "${ANDROID_HOME:-}/platform-tools/adb" \
        "${ANDROID_SDK_ROOT:-}/platform-tools/adb" \
        "$HOME/Android/Sdk/platform-tools/adb" \
        /mnt/c/platform-tools/adb.exe \
        /mnt/d/platform-tools/adb.exe; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

adb_bin=${ADB_BIN:-}
adb_cmd=()
if [[ -n "${S:-}" ]]; then
    adb_cmd+=(-s "$S")
fi
if [[ -n "${H:-}" ]]; then
    adb_cmd+=(-H "$H")
fi

log() {
    printf '\n==> %s\n' "$*"
}

require_file() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        printf 'missing required file: %s\n' "$path" >&2
        exit 1
    fi
}

require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        printf 'missing required command: %s\n' "$cmd" >&2
        exit 1
    fi
}

adb_shell() {
    "${adb_cmd[@]}" shell "$@"
}

adb_push_dir() {
    local src=$1
    local dst_parent=$2
    "${adb_cmd[@]}" push "$src" "$dst_parent/"
}

ensure_presets() {
    if [[ ! -f CMakeUserPresets.json && -f docs/backend/snapdragon/CMakeUserPresets.json ]]; then
        log "Copying Snapdragon CMakeUserPresets.json"
        cp docs/backend/snapdragon/CMakeUserPresets.json CMakeUserPresets.json
    fi
}

configure_build() {
    log "Configuring $preset -> $build_dir"
    cmake --preset "$preset" -B "$build_dir" \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        ${CMAKE_EXTRA_ARGS:-}
}

build_package() {
    log "Building libraries and executables"
    cmake --build "$build_dir" -j "$jobs"

    log "Building Hexagon HTP skels required by install"
    cmake --build "$build_dir" --target htp-v73 htp-v75 htp-v79 htp-v81 -j "$jobs"

    log "Installing package to $pkg_dir"
    rm -rf "$pkg_dir"
    cmake --install "$build_dir" --prefix "$pkg_dir"

    if [[ ! -x "$pkg_dir/bin/$cli_bin" ]]; then
        printf 'installed CLI not found: %s\n' "$pkg_dir/bin/$cli_bin" >&2
        printf 'available bin entries:\n' >&2
        find "$pkg_dir/bin" -maxdepth 1 -type f -printf '  %f\n' 2>/dev/null >&2 || true
        exit 1
    fi
}

push_to_device() {
    log "Preparing device directories"
    adb_shell "mkdir -p '$remote_root' '$remote_gguf' && rm -rf '$remote_pkg'"

    log "Pushing package to $remote_root"
    adb_push_dir "$pkg_dir" "$remote_root"

    if [[ "${SKIP_MODEL_PUSH:-0}" == "1" ]]; then
        log "Skipping GGUF model push"
    else
        log "Pushing GGUF models to $remote_gguf"
        "${adb_cmd[@]}" push "$llm_model" "$remote_gguf/"
        "${adb_cmd[@]}" push "$vit_model" "$remote_gguf/"
        "${adb_cmd[@]}" push "$dit_model" "$remote_gguf/"
    fi
}

run_on_device() {
    local llm_name vit_name dit_name
    llm_name=$(basename -- "$llm_model")
    vit_name=$(basename -- "$vit_model")
    dit_name=$(basename -- "$dit_model")

    log "Running $cli_bin on device"
    adb_shell "cd '$remote_pkg' && \
        chmod -R 755 bin lib && \
        export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib && \
        ./bin/$cli_bin \
            --llm '$remote_gguf/$llm_name' \
            --vit '$remote_gguf/$vit_name' \
            --dit '$remote_gguf/$dit_name' \
            --backend '$backend' \
            $run_args"
}

usage() {
    cat <<'EOF'
Usage:
  scripts/snapdragon/adb/run-internvla.sh

Environment overrides:
  PRESET       CMake preset (default: arm64-android-snapdragon-release)
  BUILD_DIR    Build directory (default: build-snapdragon)
  PKG_DIR      Installed package directory (default: $BUILD_DIR/pkg-snapdragon/llama.cpp)
  REMOTE_ROOT  Device root dir (default: /data/local/tmp)
  CLI_BIN      CLI binary name (default: llama-internvla-cli)
    BACKEND      Runtime backend (default: HTP0; Snapdragon Hexagon backend device)
  LLM_MODEL    Local LLM GGUF path
  VIT_MODEL    Local mmproj/VIT GGUF path
  DIT_MODEL    Local InternVLA DiT bundle GGUF path
  RUN_ARGS     Extra/runtime args (default: --dry-run)
  S            adb serial, passed as adb -s
  H            adb host, passed as adb -H
  SKIP_BUILD   Set to 1 to skip configure/build/install
  SKIP_PUSH    Set to 1 to skip adb push
    SKIP_MODEL_PUSH Set to 1 to push package only, assuming models are already on device
  SKIP_RUN     Set to 1 to skip device execution
  CMAKE_EXTRA_ARGS  Extra args appended to cmake configure

Examples:
  RUN_ARGS="--dry-run --dump" S=DEVICE_ID scripts/snapdragon/adb/run-internvla.sh
  SKIP_BUILD=1 RUN_ARGS="--image /data/local/tmp/img.jpg --instruction test" scripts/snapdragon/adb/run-internvla.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

require_cmd cmake
if [[ "${SKIP_PUSH:-0}" != "1" || "${SKIP_RUN:-0}" != "1" ]]; then
    if ! adb_bin=$(find_adb); then
        printf 'missing required command: adb. Set ADB=/path/to/adb or add adb to PATH.\n' >&2
        exit 1
    fi
    adb_cmd=("$adb_bin" "${adb_cmd[@]}")
else
    adb_cmd=(adb "${adb_cmd[@]}")
fi
require_file "$llm_model"
require_file "$vit_model"
require_file "$dit_model"
ensure_presets

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    configure_build
    build_package
else
    log "Skipping build/install"
    require_file "$pkg_dir/bin/$cli_bin"
fi

if [[ "${SKIP_PUSH:-0}" != "1" ]]; then
    push_to_device
else
    log "Skipping adb push"
fi

if [[ "${SKIP_RUN:-0}" != "1" ]]; then
    run_on_device
else
    log "Skipping device run"
fi
