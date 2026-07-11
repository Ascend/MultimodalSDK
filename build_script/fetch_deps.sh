#!/bin/bash
# -------------------------------------------------------------------------
#  Fetch build dependencies when they are not present locally.
# -------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

SKIP_FETCH=0
WITH_TEST_DEPS=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Fetch MultimodalSDK build dependencies (opensource, makeself, acc_data 3rdparty).

Options:
  --skip-fetch       Skip all downloads (offline / pre-cached builds)
  --with-test-deps   Also fetch googletest (required for build_merge.sh test)
  -h, --help         Show this help message

To remove dependencies and build outputs:
  bash build_script/clean.sh
  bash build_script/build_merge.sh clean
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fetch)
            SKIP_FETCH=1
            shift
            ;;
        --with-test-deps)
            WITH_TEST_DEPS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

check_prerequisites() {
    if ! command -v swig >/dev/null 2>&1; then
        echo "[ERROR] SWIG is required to build the AccSDK Python bindings." >&2
        echo "[ERROR] Install it with: Ubuntu/Debian: sudo apt-get install -y swig; openEuler: sudo yum install -y swig" >&2
        exit 1
    fi

    if [ "${SKIP_FETCH}" -eq 1 ]; then
        return 0
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "[ERROR] git is required to fetch dependencies." >&2
        exit 1
    fi

    if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] wget or curl is required to fetch dependencies." >&2
        exit 1
    fi
}

download_file() {
    local url=$1
    local output=$2

    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${output}" "${url}"
    else
        curl -fsSL -o "${output}" "${url}"
    fi
}

clone_repo() {
    local repo=$1
    local dest=$2
    local branch=$3

    echo "[INFO] Cloning ${repo} (${branch}) into ${dest}..."
    rm -rf "${dest}"
    git clone --depth 1 -b "${branch}" "${repo}" "${dest}"
}

fetch_opensource() {
    local tarball="${ROOT_DIR}/AccSDK/opensource.tar.gz"
    local url="${OPENSOURCE_URL:-https://mindcluster.obs.cn-north-4.myhuaweicloud.com/opensource.tar.gz}"

    if [ -f "${tarball}" ]; then
        echo "[INFO] opensource.tar.gz already present at ${tarball}"
        return 0
    fi

    echo "[INFO] Downloading opensource.tar.gz..."
    download_file "${url}" "${tarball}"
    echo "[INFO] opensource.tar.gz downloaded to ${tarball}"
}

fetch_makeself() {
    local makeself_dir="${ROOT_DIR}/makeself"
    local repo="${MAKESELF_REPO:-https://gitcode.com/gh_mirrors/ma/makeself.git}"
    local branch="${MAKESELF_VERSION:-release-2.5.0}"

    if [ -f "${makeself_dir}/makeself.sh" ]; then
        echo "[INFO] makeself already present at ${makeself_dir}"
        return 0
    fi

    clone_repo "${repo}" "${makeself_dir}" "${branch}"
    echo "[INFO] makeself installed to ${makeself_dir}"
}

fetch_makeself_patch() {
    local patch_dir="${ROOT_DIR}/makeself_patch"
    local patch_file="${patch_dir}/makeself-2.5.0.patch"
    local repo="${MAKESELF_PATCH_REPO:-https://gitcode.com/cann-src-third-party/makeself.git}"
    local branch="${MAKESELF_PATCH_VERSION:-v2.5.0.x}"

    if [ -f "${patch_file}" ]; then
        echo "[INFO] makeself patch already present at ${patch_file}"
        return 0
    fi

    clone_repo "${repo}" "${patch_dir}" "${branch}"
    echo "[INFO] makeself patch installed to ${patch_dir}"
}

fetch_acc_data_3rdparty() {
    local fetch_script="${ROOT_DIR}/AccSDK/build_script/fetch_acc_data_3rdparty.sh"

    if [ ! -f "${fetch_script}" ]; then
        echo "[ERROR] acc_data fetch script not found: ${fetch_script}" >&2
        exit 1
    fi

    if [ "${WITH_TEST_DEPS}" -eq 1 ]; then
        export FETCH_GTEST=1
    fi

    chmod +x "${fetch_script}"
    bash "${fetch_script}"
}

main() {
    check_prerequisites

    if [ "${SKIP_FETCH}" -eq 1 ]; then
        echo "[INFO] Skipping dependency fetch (--skip-fetch)"
        return 0
    fi

    echo "[INFO] Fetching MultimodalSDK build dependencies..."
    fetch_opensource
    fetch_makeself
    fetch_makeself_patch
    fetch_acc_data_3rdparty
    echo "[INFO] All dependencies are ready."
}

main
