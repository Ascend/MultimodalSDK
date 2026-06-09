#!/bin/bash
# -------------------------------------------------------------------------
#  Fetch acc_data third-party sources when git submodules are not present.
# -------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ACC_DATA_ROOT=$(cd "${SCRIPT_DIR}/../acc_data" && pwd)
THIRDPARTY_DIR="${ACC_DATA_ROOT}/3rdparty"

PYBIND11_DIR="${THIRDPARTY_DIR}/pybind/pybind11"
PYBIND11_REPO="${PYBIND11_REPO:-https://gitcode.com/GitHub_Trending/py/pybind11.git}"
PYBIND11_VERSION="${PYBIND11_VERSION:-v2.13.6}"
PYBIND11_URL="https://github.com/pybind/pybind11/archive/refs/tags/${PYBIND11_VERSION}.tar.gz"

GTEST_DIR="${THIRDPARTY_DIR}/gtest/googletest"
GTEST_REPO="${GTEST_REPO:-https://gitcode.com/GitHub_Trending/go/googletest.git}"
GTEST_VERSION="${GTEST_VERSION:-release-1.11.0}"
GTEST_URL="https://github.com/google/googletest/archive/refs/heads/${GTEST_VERSION}.tar.gz"

clone_repo() {
    local repo=$1
    local dest=$2
    local branch=$3

    echo "[INFO] Cloning ${repo} (${branch}) into ${dest}..."
    rm -rf "${dest}"
    git clone --depth 1 -b "${branch}" "${repo}" "${dest}"
}

download_tarball() {
    local url=$1
    local dest=$2
    local strip_prefix=$3
    local tmp_dir

    tmp_dir=$(mktemp -d)
    trap 'rm -rf "${tmp_dir}"' RETURN

    echo "[INFO] Downloading from ${url}..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${tmp_dir}/archive.tar.gz"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${tmp_dir}/archive.tar.gz" "${url}"
    else
        echo "[ERROR] curl or wget is required to download ${url}" >&2
        return 1
    fi

    tar -xzf "${tmp_dir}/archive.tar.gz" -C "${tmp_dir}"
    rm -rf "${dest}"
    mv "${tmp_dir}/${strip_prefix}"* "${dest}"
}

fetch_pybind11() {
    if [ -f "${PYBIND11_DIR}/CMakeLists.txt" ]; then
        echo "[INFO] pybind11 already present at ${PYBIND11_DIR}"
        return 0
    fi

    echo "[INFO] Fetching pybind11 ${PYBIND11_VERSION}..."
    mkdir -p "${THIRDPARTY_DIR}/pybind"

    if command -v git >/dev/null 2>&1; then
        if clone_repo "${PYBIND11_REPO}" "${PYBIND11_DIR}" "${PYBIND11_VERSION}"; then
            echo "[INFO] pybind11 installed to ${PYBIND11_DIR}"
            return 0
        fi
        echo "[WARN] git clone failed, falling back to tarball download." >&2
    fi

    download_tarball "${PYBIND11_URL}" "${PYBIND11_DIR}" "pybind11-"
    echo "[INFO] pybind11 installed to ${PYBIND11_DIR}"
}

fetch_googletest() {
    if [ "${FETCH_GTEST:-0}" != "1" ]; then
        return 0
    fi

    if [ -f "${GTEST_DIR}/CMakeLists.txt" ]; then
        echo "[INFO] googletest already present at ${GTEST_DIR}"
        return 0
    fi

    echo "[INFO] Fetching googletest ${GTEST_VERSION}..."
    mkdir -p "${THIRDPARTY_DIR}/gtest"

    if command -v git >/dev/null 2>&1; then
        if clone_repo "${GTEST_REPO}" "${GTEST_DIR}" "${GTEST_VERSION}"; then
            echo "[INFO] googletest installed to ${GTEST_DIR}"
            return 0
        fi
        echo "[WARN] git clone failed, falling back to tarball download." >&2
    fi

    download_tarball "${GTEST_URL}" "${GTEST_DIR}" "googletest-"
    echo "[INFO] googletest installed to ${GTEST_DIR}"
}

fetch_pybind11
fetch_googletest
