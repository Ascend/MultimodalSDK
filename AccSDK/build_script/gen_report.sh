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

WORK_DIR=$(pwd)
OUTPUT_DIR=${WORK_DIR}/coverage-report
PROJECT_ROOT=$(realpath "${WORK_DIR}/..")

# Step 1: 收集覆盖率信息，忽略 mismatch 错误
lcov --rc lcov_branch_coverage=1 --capture --directory "${PROJECT_ROOT}" --filter branch --output-file all.info --ignore-errors mismatch>/dev/null

# Step 2: 只保留项目相关覆盖率信息（提取匹配路径）
lcov --rc lcov_branch_coverage=1 -e all.info "${PROJECT_ROOT}/*" --filter branch -o project.info

# Step 3: 去除 test 文件夹下的信息
lcov --rc lcov_branch_coverage=1 -r project.info "${PROJECT_ROOT}/test/*" "${PROJECT_ROOT}/build/*" "${PROJECT_ROOT}/acc_data/*" "/usr/include/c++/*" --filter branch -o coverage.info

# Step 4: 生成 HTML 报告
genhtml --rc genhtml_branch_coverage=1 coverage.info --filter branch --output-directory "$OUTPUT_DIR"