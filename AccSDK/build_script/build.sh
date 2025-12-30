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
# Description: Build script of ACC SDK.
# Author: ACC SDK
# Create: 2025
# History: NA
set -e

rm -rf ../build/*

# 构建信息环境变量（可以提前 export 也可脚本内赋值）
SCRIPT_DIR_ACC_SDK=$(cd "$(dirname "$0")"; pwd)
ACC_SDK_ROOT_DIR=$(cd "${SCRIPT_DIR_ACC_SDK}/.."; pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/opt/buildtools/python-3.11.4/bin:$PATH"
rm -rf "${ACC_SDK_ROOT_DIR}/build/*"

ACC_SDK_DIR="$SCRIPT_DIR_ACC_SDK"
ACC_BUILD_SCRIPT=$(basename "$0")
ACC_BUILD_TYPE=${1:-Release}

chmod +x "${SCRIPT_DIR_ACC_SDK}/build_opensource.sh"
. "${SCRIPT_DIR_ACC_SDK}/build_opensource.sh" || { echo "opensource compile failed"; exit 1; }

chmod +x "${SCRIPT_DIR_ACC_SDK}/build_acc_data.sh"
. "${SCRIPT_DIR_ACC_SDK}/build_acc_data.sh" || { echo "acc_data compile failed"; exit 1; }

# 输出函数
print_build_start() {
  echo "==============BUILDing ACCSDK REPO=============="
}

print_build_success() {
  echo "==============BUILD ACCSDK REPO SUCCESS=============="
}

print_build_failure() {
  echo "==============BUILD ACCSDK REPO FAILED=============="
}

# 捕捉错误退出
trap 'print_build_failure; exit 1' ERR

# 构建变量
BUILD_IMAGE="${IMAGE:-ON}"
BUILD_VIDEO="${VIDEO:-ON}"

# 打印开始构建信息
print_build_start

# cmake 配置
if [[ x"$1" == x"test" ]]; then
  echo ">>> Running cmake in GTEST mode"
  cd ${ACC_SDK_ROOT_DIR}/acc_data/3rdparty/gtest/googletest
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
  make -j
  make install

  cmake -S "${ACC_SDK_ROOT_DIR}" -B"${ACC_SDK_ROOT_DIR}/build" \
    -DIMAGE="${BUILD_IMAGE}" \
    -DVIDEO="${BUILD_VIDEO}" \
    -DBUILD_TESTS=ON
else
  echo ">>> Running cmake in Release mode"
  cmake -S "${ACC_SDK_ROOT_DIR}" -B"${ACC_SDK_ROOT_DIR}/build" \
    -DIMAGE="${BUILD_IMAGE}" \
    -DVIDEO="${BUILD_VIDEO}"
fi

# 编译并安装
echo ">>> Running make..."
make -C "${ACC_SDK_ROOT_DIR}/build" -j || { echo "make failed"; exit 1; }

echo ">>> Running make install..."
(cd "${ACC_SDK_ROOT_DIR}/build" && make install)

# 拷贝构建产物
echo ">>> Copying build artifacts..."

OUTPUT_DIR="${ACC_SDK_ROOT_DIR}/output"
SECUREC_SRC_DIR="${ACC_SDK_ROOT_DIR}/acc_data/output/3rdparty/securec/lib"
ACC_INCLUDE_SRC_DIR="${ACC_SDK_ROOT_DIR}/acc_data/src/cpp/interface"
BUILD_LIB_DIR="${ACC_SDK_ROOT_DIR}/build/lib"

# 创建目标目录"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/include/acc_data"
mkdir -p "${OUTPUT_DIR}/include/acc_sdk"
mkdir -p "${OUTPUT_DIR}/lib"
mkdir -p "${OUTPUT_DIR}/opensource/libjpeg-turbo"
mkdir -p "${OUTPUT_DIR}/opensource/FFmpeg"

# 拷贝 AccData 头文件
if compgen -G "${ACC_INCLUDE_SRC_DIR}/*.h" > /dev/null; then
  cp -v "${ACC_INCLUDE_SRC_DIR}"/*.h "${OUTPUT_DIR}/include/acc_data/"
else
  echo "Warning: No header files found in ${ACC_INCLUDE_SRC_DIR}"
  exit 1
fi

if [ -d "${ACC_SDK_ROOT_DIR}/include" ] && [ "$(ls -A "${ACC_SDK_ROOT_DIR}/include")" ]; then
    echo "Copying headers from ${ACC_SDK_ROOT_DIR}/output to ${OUTPUT_DIR}/include/acc_sdk/"
    cp -rv "${ACC_SDK_ROOT_DIR}/include/." "${OUTPUT_DIR}/include/acc_sdk/"
else
    echo "Warning: No files or directories found in ${ACC_SDK_ROOT_DIR}/output"
    exit 1
fi

# 拷贝 build 产物 .so 文件
if compgen -G "${BUILD_LIB_DIR}/*.so" > /dev/null; then
  cp -v "${BUILD_LIB_DIR}"/*.so "${OUTPUT_DIR}/lib/"
else
  echo "Warning: No .so files found in ${BUILD_LIB_DIR}"
  exit 1
fi

if compgen -G "${BUILD_LIB_DIR}/../source/py/*.so" > /dev/null; then
  cp -v "${BUILD_LIB_DIR}/../source/py"/*.so "${OUTPUT_DIR}/"
else
  echo "Warning: No .so files found in ${BUILD_LIB_DIR}/../source/py"
  exit 1
fi

if compgen -G "${BUILD_LIB_DIR}/../source/py/*.py" > /dev/null; then
  cp -v "${BUILD_LIB_DIR}/../source/py"/*.py "${OUTPUT_DIR}/"
else
  echo "Warning: No .py files found in ${BUILD_LIB_DIR}/../source/py"
  exit 1
fi

copy_opensource_outputs() {
  local component=$1
  local src_base="${ACC_SDK_ROOT_DIR}/opensource/${component}"
  local dst_base="${OUTPUT_DIR}/opensource/${component}"

  mkdir -p "${dst_base}"

  for subdir in lib include share bin; do
    if compgen -G "${src_base}/${subdir}/*" > /dev/null; then
      mkdir -p "${dst_base}/${subdir}"
      cp -rv "${src_base}/${subdir}/"* "${dst_base}/${subdir}/"
    else
      echo "Warning: No files found in ${src_base}/${subdir}"
    fi
  done
}

copy_opensource_outputs libjpeg-turbo
copy_opensource_outputs FFmpeg

TAR_NAME="acc_sdk_linux-aarch64.tar.gz"
tar -czvf "${OUTPUT_DIR}/${TAR_NAME}" --exclude="${TAR_NAME}" -C "${OUTPUT_DIR}" $(ls -A "${OUTPUT_DIR}")

echo "tar success：${OUTPUT_DIR}/${TAR_NAME}"


# 成功完成
print_build_success
