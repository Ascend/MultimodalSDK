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
# Used to build project files

set -e
export CMAKE_BUILD_TYPE="$1"
USING_COVERAGE="$2"
USING_XSAN="$3"
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
source "${CURRENT_PATH}/build_env.sh"

CPU_TYPE="$(arch)"
echo "Build ${ACCDATA_BUILD_PATH} Type=${CMAKE_BUILD_TYPE}"
if [ ! -d "${ACCDATA_BUILD_PATH}" ]; then
    mkdir -p "${ACCDATA_BUILD_PATH}"
else
    [ -d "${ACCDATA_BUILD_PATH}" ] && rm -rf "${ACCDATA_BUILD_PATH}"
    mkdir -p "${ACCDATA_BUILD_PATH}"
fi

[ -d "${OUTPUT_PATH}/AccData" ] && rm -rf "${OUTPUT_PATH}/AccData"
mkdir -p "${OUTPUT_PATH}/AccData"

# use whl for tmp
python3 setup.py bdist_wheel

echo "ACCDATA_ROOT_PATH is ${ACCDATA_ROOT_PATH}"
cp -rf "${ACCDATA_ROOT_PATH}/src/cpp/interface" "${ACCDATA_ROOT_PATH}/output/AccData/include"