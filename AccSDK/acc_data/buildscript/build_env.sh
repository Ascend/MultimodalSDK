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
CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
export ACCDATA_ROOT_PATH=$(cd "${CURRENT_PATH}/.."; pwd)
export OUTPUT_PATH="${ACCDATA_ROOT_PATH}/output"

export ACCDATA_SRC_PATH="${ACCDATA_ROOT_PATH}/src"
export ACCDATA_TEST_PATH="${ACCDATA_ROOT_PATH}/tests"
export ACCDATA_THIRDPART_PATH="${ACCDATA_ROOT_PATH}/3rdparty"
export ACCDATA_BUILD_PATH="${ACCDATA_ROOT_PATH}/build"
export CPU_TYPE=$(arch)
export INSTALL_PATH="${ACCDATA_ROOT_PATH}/install"

BUILD_CPUS=$(grep processor /proc/cpuinfo | wc -l)
if [[ ${BUILD_CPUS} -gt 2 ]];then
    BUILD_CPUS=$((BUILD_CPUS/2))
fi
# the parameter for N in command `make -j N`, we use half of the available CPU by default
export CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_CPUS}
export ACCDATA_ENABLE_TRACER=0
