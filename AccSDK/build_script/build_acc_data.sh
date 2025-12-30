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
# Description: Build script of acc_data.
# Author: ACC SDK
# Create: 2025
# History: NA
set -e

# ----------------------
# 配置区域
# ----------------------

# acc_data相对路径为 ./acc_data
SCRIPT_DIR_ACC_DATA=$(cd "$(dirname "$0")"; pwd)
ACC_DATA_ROOT_DIR=$(cd "${SCRIPT_DIR_ACC_DATA}/.."; pwd)
ACC_DATA_NAME="acc_data"
ACC_DATA_REPO_DIR="${ACC_DATA_ROOT_DIR}/${ACC_DATA_NAME}"
# acc_data仓库的构建脚本路径
ACC_DATA_BUILD_SCRIPT="${ACC_DATA_REPO_DIR}/build.sh"
# 构建
ACC_DATA_BUILD_TYPE="release"
# 清理
ACC_DATA_CLEAN_TYPE="clean"

print_build_start() {
  echo "==============BUILDing ACC_DATA REPO=============="
}

print_build_success() {
  echo "==============BUILD ACC_DATA REPO SUCCESS=============="
}

print_build_failure() {
  echo "==============BUILD ACC_DATA REPO FAILED=============="
}

echo "[INFO] Starting build script for ACC_DATA repo: ${ACC_DATA_REPO_DIR}"
print_build_start


# ----------------------
# 基础检查
# ----------------------

if [ ! -d "${ACC_DATA_REPO_DIR}" ]; then
    echo "[ERROR] ACC_DATA repo directory does not exist: ${ACC_DATA_REPO_DIR}" >&2
    print_build_failure
    exit 1
fi

# ----------------------
# ACC_DATA仓cpp层CMakeLists.txt修改，替换 add_library(_accdata SHARED) -> STATIC
# ----------------------

echo "[INFO] Modifying CMakeLists.txt to set _accdata as STATIC library..."

ACC_DATA_CMAKE_FILE="${ACC_DATA_REPO_DIR}/src/cpp/CMakeLists.txt"

if [ ! -f "$ACC_DATA_CMAKE_FILE" ]; then
    echo "[ERROR] CMakeLists.txt not found in B repo: $ACC_DATA_CMAKE_FILE" >&2
    print_build_failure
    exit 1
fi

# 修改cmakefile前备份
cp "$ACC_DATA_CMAKE_FILE" "${ACC_DATA_CMAKE_FILE}.bak"

# 替换，兼容可能有空格的写法
sed -i -E 's/add_library\(\s*_accdata\s+SHARED\s*\)/add_library(_accdata STATIC)/g' "$ACC_DATA_CMAKE_FILE"
sed -i -E '/install\s*\(\s*TARGETS\s+_accdata/,/\)/ s/LIBRARY DESTINATION/ARCHIVE DESTINATION/' "$ACC_DATA_CMAKE_FILE"
sed -i '/add_link_options($<$<COMPILE_LANGUAGE:C,CXX>:-O3>)/a add_link_options($<$<COMPILE_LANGUAGE:C,CXX>:-flto=8>)' "${ACC_DATA_REPO_DIR}/CMakeLists.txt"

echo "[INFO] Modification done."

cd - >/dev/null

# ----------------------
# 执行 clean
# ----------------------

echo "[INFO] Running clean step for ACCDATA repo..."
if ! bash "${ACC_DATA_BUILD_SCRIPT}" -t "${ACC_DATA_CLEAN_TYPE}"; then
    echo "[ERROR] Clean step failed." >&2
    print_build_failure
    exit 1
fi

# ----------------------
# 执行 build
# ----------------------

echo "[INFO] Running build step for ACCDATA repo with type: ${ACC_DATA_BUILD_TYPE}..."
if bash "${ACC_DATA_BUILD_SCRIPT}" "${ACC_DATA_BUILD_TYPE}"; then
    echo "[INFO] Build succeeded."
    print_build_success
    return 0
else
    echo "[ERROR] Build failed." >&2
    print_build_failure
    exit 1
fi

