#!/bin/bash
# -------------------------------------------------------------------------
#  Remove third-party dependencies, build artifacts, and output products.
# -------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

usage() {
    cat <<EOF
Usage: $(basename "$0")

Remove all fetched third-party dependencies, build intermediate files,
and packaging outputs for MultimodalSDK.

Also invoked by: bash build_script/build_merge.sh clean

This script does not require CANN environment.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    echo "[ERROR] Unknown option: $1" >&2
    usage >&2
    exit 1
fi

remove_path() {
    local path=$1
    if [ -e "${path}" ]; then
        rm -rf "${path}"
        echo "[INFO] Removed ${path}"
    fi
}

restore_cmake_from_backup() {
    local cmake_file=$1
    if [ -f "${cmake_file}.bak" ]; then
        mv -f "${cmake_file}.bak" "${cmake_file}"
        echo "[INFO] Restored ${cmake_file} from backup"
    fi
}

restore_tracked_cmake_files() {
    local acc_data_root="${ROOT_DIR}/AccSDK/acc_data"
    local files=(
        "${acc_data_root}/src/cpp/CMakeLists.txt"
        "${acc_data_root}/CMakeLists.txt"
    )

    for cmake_file in "${files[@]}"; do
        restore_cmake_from_backup "${cmake_file}"
    done

    if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        for cmake_file in "${files[@]}"; do
            if [ -f "${cmake_file}" ]; then
                git -C "${ROOT_DIR}" checkout -- "${cmake_file}" 2>/dev/null || true
            fi
        done
        echo "[INFO] Restored tracked acc_data CMake files via git"
    fi
}

run_acc_data_clean() {
    local acc_data_build="${ROOT_DIR}/AccSDK/acc_data/build.sh"
    if [ -f "${acc_data_build}" ]; then
        echo "[INFO] Running acc_data clean..."
        bash "${acc_data_build}" -t clean
    fi
}

clean_third_party_deps() {
    echo "[INFO] Removing third-party dependencies..."
    remove_path "${ROOT_DIR}/makeself"
    remove_path "${ROOT_DIR}/makeself_patch"
    remove_path "${ROOT_DIR}/AccSDK/opensource.tar.gz"
    remove_path "${ROOT_DIR}/AccSDK/opensource"
    remove_path "${ROOT_DIR}/AccSDK/acc_data/3rdparty/pybind/pybind11"
    remove_path "${ROOT_DIR}/AccSDK/acc_data/3rdparty/gtest/googletest"
}

clean_build_artifacts() {
    echo "[INFO] Removing build intermediate files..."
    run_acc_data_clean
    remove_path "${ROOT_DIR}/AccSDK/build"
    remove_path "${ROOT_DIR}/MultimodalSDK/build"
    remove_path "${ROOT_DIR}/MultimodalSDK/dist"
    remove_path "${ROOT_DIR}/AccSDK/acc_data/dist"
    remove_path "${ROOT_DIR}/AccSDK/acc_data/build"
    remove_path "${ROOT_DIR}/AccSDK/acc_data/output"
    remove_path "${ROOT_DIR}/AccSDK/acc_data/install"
    remove_path "${ROOT_DIR}/lcov-2.0"
    remove_path "${ROOT_DIR}/lcov-2.0.tar.gz"

    find "${ROOT_DIR}/MultimodalSDK/source" -maxdepth 1 -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    restore_tracked_cmake_files
}

clean_outputs() {
    echo "[INFO] Removing build outputs..."
    remove_path "${ROOT_DIR}/AccSDK/output"
    remove_path "${ROOT_DIR}/MultimodalSDK/output"
    remove_path "${ROOT_DIR}/output"
    remove_path "${ROOT_DIR}/AccSDK/build_script/coverage-report"
    remove_path "${ROOT_DIR}/MultimodalSDK/build_script/coverage"

    find "${ROOT_DIR}" -maxdepth 3 -name "*.run" -type f -delete 2>/dev/null || true
    find "${ROOT_DIR}/AccSDK/build_script" -maxdepth 1 -name "*.info" -type f -delete 2>/dev/null || true
}

main() {
    echo "[INFO] Cleaning MultimodalSDK workspace at ${ROOT_DIR}..."
    clean_outputs
    clean_build_artifacts
    clean_third_party_deps
    echo "[INFO] Clean completed."
}

main
