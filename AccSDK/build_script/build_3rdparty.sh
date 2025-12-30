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
# Description: Build script of ACC SDK opensource.
# Author: ACC SDK
# Create: 2025
# History: NA
set -e

# ----------------------
# 配置区域
# ----------------------

# B 仓库目录，假设相对路径为 ./3rdparty/acc_data
SCRIPT_DIR_3RD=$(cd "$(dirname "$0")"; pwd)
A_ROOT_DIR=$(cd "${SCRIPT_DIR_3RD}/.."; pwd)
B_REPO_NAME="acc_data"
B_REPO_DIR="${A_ROOT_DIR}/3rdparty/${B_REPO_NAME}"
# B 仓库的构建脚本路径
B_BUILD_SCRIPT="${B_REPO_DIR}/build.sh"
# 构建
B_BUILD_TYPE="release"
# 清理
B_CLEAN_TYPE="clean"

print_build_start() {
  echo "==============BUILDing ACCDATA REPO=============="
}

print_build_success() {
  echo "==============BUILD ACCDATA REPO SUCCESS=============="
}

print_build_failure() {
  echo "==============BUILD ACCDATA REPO FAILED=============="
}

echo "[INFO] Starting build script for ACCDATA repo: ${B_REPO_DIR}"
print_build_start


# ----------------------
# 基础检查
# ----------------------

if [ ! -d "${B_REPO_DIR}" ]; then
    echo "[ERROR] ACCDATA repo directory does not exist: ${B_REPO_DIR}" >&2
    print_build_failure
    exit 1
fi

if [ ! -f "${B_BUILD_SCRIPT}" ]; then
    echo "[ERROR] Build script not found in ACCDATA repo: ${B_BUILD_SCRIPT}" >&2
    print_build_failure
    exit 1
fi

# ----------------------
# B仓顶层CMakeLists.txt修改，替换 add_library(_accdata SHARED) -> STATIC
# ----------------------

echo "[INFO] Modifying B repo CMakeLists.txt to set _accdata as STATIC library..."

B_CMAKE_FILE="${B_REPO_DIR}/src/cpp/CMakeLists.txt"

if [ ! -f "$B_CMAKE_FILE" ]; then
    echo "[ERROR] CMakeLists.txt not found in B repo: $B_CMAKE_FILE" >&2
    print_build_failure
    exit 1
fi

# 备份
cp "$B_CMAKE_FILE" "${B_CMAKE_FILE}.bak"

# 替换，兼容可能有空格的写法
sed -i -E 's/add_library\(\s*_accdata\s+SHARED\s*\)/add_library(_accdata STATIC)/g' "$B_CMAKE_FILE"
sed -i -E '/install\s*\(\s*TARGETS\s+_accdata/,/\)/ s/LIBRARY DESTINATION/ARCHIVE DESTINATION/' "$B_CMAKE_FILE"
sed -i '/add_link_options($<$<COMPILE_LANGUAGE:C,CXX>:-O3>)/a add_link_options($<$<COMPILE_LANGUAGE:C,CXX>:-flto=8>)' "${B_REPO_DIR}/CMakeLists.txt"

echo "[INFO] Modification done."

cd - >/dev/null

# ----------------------
# 执行 clean
# ----------------------

echo "[INFO] Running clean step for ACCDATA repo..."
if ! bash "${B_BUILD_SCRIPT}" -t "${B_CLEAN_TYPE}"; then
    echo "[ERROR] Clean step failed." >&2
    print_build_failure
    exit 1
fi

# ----------------------
# 执行 build
# ----------------------

echo "[INFO] Running build step for ACCDATA repo with type: ${B_BUILD_TYPE}..."
if bash "${B_BUILD_SCRIPT}" "${B_BUILD_TYPE}"; then
    echo "[INFO] Build succeeded."
    print_build_success
    return 0
else
    echo "[ERROR] Build failed." >&2
    print_build_failure
    exit 1
fi