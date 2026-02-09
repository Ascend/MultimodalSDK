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
# Description: SDK build script.
# Author: Multimodal SDK
# Create: 2025
# History: NA
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
A_ROOT_DIR=$(cd "$SCRIPT_DIR/.."; pwd)

B_REPO_DIR="${A_ROOT_DIR}/../AccSDK"
B_BUILD_SCRIPT="${B_REPO_DIR}/build_script/build.sh"
PACKAGE_DIR="${A_ROOT_DIR}/source/mm/acc/_impl"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/opt/buildtools/python-3.11.4/bin:$PATH"
VERSION="dev"
PARSED_BUILD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        *)
            PARSED_BUILD_ARGS+=("$1")
            shift
            ;;
    esac
done

pip3 install pillow==12.0.0
pip3 install torchvision==0.24.1

BUILD_ARGS=("${PARSED_BUILD_ARGS[@]}")
echo "[INFO] Cleaning build, dist, egg-info..."
rm -rf "${A_ROOT_DIR}/build" "${A_ROOT_DIR}/dist"
rm -rf "${A_ROOT_DIR}"/source/*.egg-info

echo "[INFO] Step 3: Copy and build whl..."

# find acc.cpython*.so
SO_SRC=$(find "${B_REPO_DIR}/output" -name "_acc.so" | head -n 1)
if [ -z "$SO_SRC" ]; then
    echo "[ERROR] File not found: _acc.so"
    exit 1
fi

mkdir -p "${PACKAGE_DIR}"
SO_DST="${PACKAGE_DIR}/$(basename "$SO_SRC")"
cp -f "$SO_SRC" "$SO_DST"
echo "[INFO] Copying .so to: ${SO_DST}"

# find acc.py
SO_SRC=$(find "${B_REPO_DIR}/output" -name "acc.py" | head -n 1)
if [ -z "$SO_SRC" ]; then
    echo "[ERROR] File not found: acc.py"
    exit 1
fi

SO_DST="${PACKAGE_DIR}/$(basename "$SO_SRC")"
cp -f "$SO_SRC" "$SO_DST"
echo "[INFO] Copying acc.py to: ${SO_DST}"

# build whl
cd "${A_ROOT_DIR}"
echo "[INFO] Executing python3 setup.py bdist_wheel..."
python3 setup.py bdist_wheel

echo "[INFO] Building done!"

OUTPUT_DIR="${A_ROOT_DIR}/output"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/lib"
mkdir -p "${OUTPUT_DIR}/script"
mkdir -p "${OUTPUT_DIR}/opensource/FFmpeg"
mkdir -p "${OUTPUT_DIR}/opensource/libjpeg-turbo"
mkdir -p "${OUTPUT_DIR}/opensource/soxr"

if compgen -G "${A_ROOT_DIR}/dist/*.whl" > /dev/null; then
  cp -v "${A_ROOT_DIR}/dist/"*.whl "${OUTPUT_DIR}/"
else
  echo "[Warning] No whl file found in ${A_ROOT_DIR}/dist"
fi

if compgen -G "${B_REPO_DIR}/output/lib/libcore.so" > /dev/null; then
  cp -v "${B_REPO_DIR}/output/lib/libcore.so" "${OUTPUT_DIR}/lib/"
else
  echo "[Warning] libcore.so not found!"
fi

cp -rf "${B_REPO_DIR}/output/opensource/FFmpeg/lib" "${OUTPUT_DIR}/opensource/FFmpeg"
rm -rf "${OUTPUT_DIR}/opensource/FFmpeg/lib/pkgconfig"

cp -rf "${B_REPO_DIR}/output/opensource/libjpeg-turbo/lib" "${OUTPUT_DIR}/opensource/libjpeg-turbo"
rm -rf "${OUTPUT_DIR}/opensource/libjpeg-turbo/lib/cmake"
rm -rf "${OUTPUT_DIR}/opensource/libjpeg-turbo/lib/pkgconfig"

cp -rf "${B_REPO_DIR}/output/opensource/soxr/lib" "${OUTPUT_DIR}/opensource/soxr"
rm -rf "${OUTPUT_DIR}/opensource/soxr/lib/pkgconfig"

cp "${A_ROOT_DIR}/script/set_env.sh" "${OUTPUT_DIR}/script/"
cp "${A_ROOT_DIR}/script/uninstall.sh" "${OUTPUT_DIR}/script/"
VERSION_MAJOR=$(echo "$VERSION" | sed -E 's/\.b[0-9]+$//')
echo -e "MindX SDK multimodal-sdk:${VERSION_MAJOR}\nmultimodal-sdk version:${VERSION}\nPlat: linux aarch64" > "${OUTPUT_DIR}/version.info"
TAR_NAME="multimodal-sdk-${VERSION_TAR}_linux-aarch64.tar.gz"
echo "[INFO] Packaging output to ${TAR_NAME}..."
tar -czvf "${OUTPUT_DIR}/${TAR_NAME}" --exclude="${TAR_NAME}" -C "${OUTPUT_DIR}" $(ls -A "${OUTPUT_DIR}")
cp "${A_ROOT_DIR}/script/help.info" "${OUTPUT_DIR}/script/"
cp "${A_ROOT_DIR}/script/install.sh" "${OUTPUT_DIR}/script/"
chmod +x "${OUTPUT_DIR}/script/install.sh"
if [[ "${BUILD_ARGS[*]}" == *"test"* ]]; then
    echo "[INFO] Building test: install whl first..."
    WHL_FILE=$(find "${A_ROOT_DIR}/dist" -name "*.whl" | head -n 1)
    if [ -z "$WHL_FILE" ]; then
        echo "[ERROR] whl package not found!"
        exit 1
    fi
    # extract file name
    WHL_NAME=$(basename "$WHL_FILE")
    PKG_NAME=${WHL_NAME%%-*}
    echo "[INFO] Uninstall WHL package: $PKG_NAME"
    pip uninstall -y "$PKG_NAME" 2>/dev/null || echo "[INFO] WHL not installed, skipping uninstall"
    echo "[INFO] Executing pytest with coverage report..."

    cd "${A_ROOT_DIR}"
    export LD_LIBRARY_PATH="${A_ROOT_DIR}/output/lib:/usr1/AccSDK/output/opensource/libjpeg-turbo/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="/usr1/AccSDK/output/opensource/FFmpeg/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="/usr1/AccSDK/output/opensource/soxr/lib:$LD_LIBRARY_PATH"
    # clean coverage first
    coverage erase
    LD_PRELOAD=/opt/buildtools/python-3.11.4/lib/python3.11/site-packages/torch.libs/libgomp-98df74fd.so.1.0.0 \
    PYTHONPATH=source \
    pytest \
    --cov=source/mm \
    --cov-report=html:build_script/coverage/html \
    --cov-report=xml:build_script/coverage/coverage.xml \
    --junit-xml=build_script/coverage/final.xml \
    --html=build_script/coverage/final.html \
    --self-contained-html \
    --cov-branch \
    -vs test/

    echo "[INFO] Coverage report generated:"
    echo "  HTML: ${A_ROOT_DIR}/build_script/coverage/htmlcov/index.html"
    echo "  XML : ${A_ROOT_DIR}/build_script/coverage/coverage.xml"
    echo "  JUnit: ${A_ROOT_DIR}/build_script/coverage/final.xml"
    echo "  HTML test report: ${A_ROOT_DIR}/build_script/coverage/final.html"

    echo "[INFO] Test done!"
fi
