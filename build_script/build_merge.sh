#!/bin/bash
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
# Description: SDK merge build script.
# Author: Multimodal SDK
# Create: 2025
# History: NA

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MERGE_BUILD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "clean" ]]; then
    chmod +x "${SCRIPT_DIR}/clean.sh"
    bash "${SCRIPT_DIR}/clean.sh"
    exit 0
fi

SDK_VERSION="${SDK_VERSION:-$(sed -n 's/^version:[[:space:]]*//p' "${MERGE_BUILD_DIR}/../ci/config/config.ini" 2>/dev/null || true)}"
SDK_VERSION="${SDK_VERSION:-0.0.0.dev}"

ASCEND_SET_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
if [ ! -f "${ASCEND_SET_ENV}" ]; then
    echo "[ERROR] CANN environment script not found: ${ASCEND_SET_ENV}" >&2
    echo "[ERROR] Please install CANN and source set_env.sh before building." >&2
    exit 1
fi
# Ascend set_env.sh appends to LD_LIBRARY_PATH/PYTHONPATH/PATH; with set -u those must exist first.
: "${LD_LIBRARY_PATH:=}"
: "${PYTHONPATH:=}"
: "${PATH:=}"
# shellcheck source=/dev/null
source "${ASCEND_SET_ENV}"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

FETCH_ARGS=()
if [[ "${1:-}" == "test" ]]; then
    FETCH_ARGS+=(--with-test-deps)
fi
chmod +x "${SCRIPT_DIR}/fetch_deps.sh"
bash "${SCRIPT_DIR}/fetch_deps.sh" "${FETCH_ARGS[@]}"

ACC_SDK_ROOT_DIR="${MERGE_BUILD_DIR}/AccSDK"
if [ ! -f "${ACC_SDK_ROOT_DIR}/opensource.tar.gz" ]; then
    echo "[ERROR] opensource.tar.gz not found in ${ACC_SDK_ROOT_DIR}" >&2
    exit 1
fi

if [ ! -d "${ACC_SDK_ROOT_DIR}/opensource/FFmpeg" ]; then
    echo "[INFO] Extracting opensource.tar.gz..."
    tar -zxf "${ACC_SDK_ROOT_DIR}/opensource.tar.gz" -C "${ACC_SDK_ROOT_DIR}"
fi

ACC_BUILD_DIR="${ACC_SDK_ROOT_DIR}/build_script"
cd "${ACC_BUILD_DIR}" || { echo "[ERROR] Cannot enter directory ${ACC_BUILD_DIR}"; exit 1; }

chmod +x build.sh
if [[ "${1:-}" == "test" ]]; then
    cd "${MERGE_BUILD_DIR}"
    NEED_LCOV=false
    if command -v lcov >/dev/null 2>&1; then
        LCOV_VER=$(lcov --version 2>/dev/null | grep -oP '[\d]+\.[\d]+' | head -1)
        if [ -n "${LCOV_VER}" ] && [ "$(printf '%s\n' "2.0" "${LCOV_VER}" | sort -V | head -1)" = "2.0" ]; then
            echo "[INFO] System lcov ${LCOV_VER} meets requirement (>=2.0), reusing."
        else
            NEED_LCOV=true
        fi
    else
        NEED_LCOV=true
    fi
    if [ "${NEED_LCOV}" = true ]; then
        if [ ! -d "${MERGE_BUILD_DIR}/lcov-2.0" ]; then
            echo "[INFO] Downloading lcov 2.0..."
            wget -q https://github.com/linux-test-project/lcov/releases/download/v2.0/lcov-2.0.tar.gz
            tar -xzf lcov-2.0.tar.gz
        fi
        cd lcov-2.0
        make install
    fi
    cd "${ACC_BUILD_DIR}"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/opensource/FFmpeg/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/opensource/libjpeg-turbo/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/opensource/soxr/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/output/lib:${LD_LIBRARY_PATH}"
    ./build.sh test || exit 1
    export GTEST_HOME="${ACC_SDK_ROOT_DIR}/acc_data/3rdparty/gtest/googletest/build/googletest"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/acc_data/3rdparty/gtest/googletest/build/lib/:${LD_LIBRARY_PATH}"
    BUILD_DIR="${ACC_SDK_ROOT_DIR}/build"
    cd "$BUILD_DIR"
    # Determine parallel level for tests: use PARALLEL_LEVEL if set, else auto-detect
    TEST_RC=""
    if [ -z "${PARALLEL_LEVEL:-}" ]; then
        # Auto-detect number of CPUs
        if command -v nproc >/dev/null 2>&1; then
            total_cpus=$(nproc) || total_cpus=""
            # Use half of available CPUs to avoid resource contention and OOM
            # Use expr for maximum shell compatibility, || fallback if expr fails
            PARALLEL_LEVEL=$(expr \( $total_cpus + 1 \) / 2) || PARALLEL_LEVEL=8
        else
            PARALLEL_LEVEL=8
        fi
    fi
    # Ensure minimum 1, reasonable maximum 16 (extra safety)
    if ! echo "$PARALLEL_LEVEL" | grep -qE '^[0-9]+$'; then
        PARALLEL_LEVEL=8
    fi
    if [ "$PARALLEL_LEVEL" -lt 1 ]; then
        PARALLEL_LEVEL=1
    fi
    if [ "$PARALLEL_LEVEL" -gt 16 ]; then
        PARALLEL_LEVEL=16
    fi
    echo ">>> Running tests with PARALLEL_LEVEL=$PARALLEL_LEVEL (half of total CPUs for resource safety)"
    # Use ctest directly for better parallel control
    # Let ctest generate unified JUnit report to avoid concurrent write to same xml by individual tests
    ctest -j "$PARALLEL_LEVEL" --output-on-failure --output-junit test_detail.xml || TEST_RC=$?
    if [ -f "$BUILD_DIR/Testing/Temporary/LastTest.log" ]; then
        cat "$BUILD_DIR/Testing/Temporary/LastTest.log"
    else
        echo "Warning: LastTest.log not found at $BUILD_DIR/Testing/Temporary/LastTest.log"
    fi
    if [ -n "${TEST_RC:-}" ]; then
        exit $TEST_RC
    fi
    cd "${SCRIPT_DIR}" && bash gen_report.sh && python3 testcases_xml_report.py ../test coverage-report
else
    ./build.sh || exit 1
fi

MULTI_SDK_ROOT_DIR="${MERGE_BUILD_DIR}/MultimodalSDK"
MULTI_BUILD_DIR="${MULTI_SDK_ROOT_DIR}/build_script"
cd "${MULTI_BUILD_DIR}" || { echo "[ERROR] Cannot enter directory ${MULTI_BUILD_DIR}"; exit 1; }

export PYTHONPATH=${MULTI_SDK_ROOT_DIR}/source/:${PYTHONPATH:-}

chmod +x build.sh
if [[ "${1:-}" == "test" ]]; then
    ./build.sh test || exit 1
else
    ./build.sh --version "${SDK_VERSION}" || exit 1
fi

PACKAGE_DIR="${MERGE_BUILD_DIR}/makeself"
PATCH_FILE="${MERGE_BUILD_DIR}/makeself_patch/makeself-2.5.0.patch"
cd "${PACKAGE_DIR}" || { echo "[ERROR] Cannot enter directory ${PACKAGE_DIR}"; exit 1; }

if [ -f ".mmsdk_patched" ] || grep -q 'PACKAGE_LOG_NAME=makeself' makeself-header.sh; then
    echo "[INFO] makeself patch already applied, skipping"
else
    if [ ! -f "${PATCH_FILE}" ]; then
        echo "[ERROR] makeself patch not found: ${PATCH_FILE}" >&2
        exit 1
    fi
    patch -p1 < "${PATCH_FILE}" || { echo "[ERROR] Patch failed"; exit 1; }
    touch .mmsdk_patched
fi

cd "${SCRIPT_DIR}" || { echo "[ERROR] Cannot enter directory ${SCRIPT_DIR}"; exit 1; }
export SDK_VERSION
chmod +x package.sh
./package.sh || { echo "[ERROR] Packaging failed"; exit 1; }

echo "Packaging completed successfully!"
