#!/bin/bash
# Used to build and run dt files
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

set -e
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)

# **************************代码构建 Begin **************************
hdt comp -i || true

ASAN_LOG_PATH=${CURRENT_PATH}/asanLog
if [ -d ${ASAN_LOG_PATH} ]; then
    rm -rf ${ASAN_LOG_PATH}
fi
mkdir -p ${ASAN_LOG_PATH}
export ASAN_OPTIONS=halt_on_error=0:log_path=${ASAN_LOG_PATH}/asan.log:detect_stack_use_after_return=1:check_initialization_order=1:verbosity=1:strict_string_checks=1:strict_init_order=1

export BUILD_MODE=fuzz
hdt build --task fuzz_test "$@"
hdt build --task cmake_test "$@"
unset BUILD_MODE
# **************************代码构建 End ****************************

# **************************测试执行 Begin **************************
hdt run --args="--gtest_output=xml:report.xml"
# **************************测试执行 End ***************************

# **************************报告生成 Begin **************************

dt_fuzz_report()
{
    FUZZ_GENERATE_DIR=${CURRENT_PATH}/fuzz/cov/gen
    if [ -d ${FUZZ_GENERATE_DIR} ]; then
        rm -rf ${FUZZ_GENERATE_DIR}
    fi
    mkdir -p ${FUZZ_GENERATE_DIR}
    find ${CURRENT_PATH}/../.. -name "*.gcda" | xargs  -i mv {} ${FUZZ_GENERATE_DIR}
    find ${CURRENT_PATH}/../.. -name "*.expand" | xargs  -i mv {} ${FUZZ_GENERATE_DIR}
    find ${CURRENT_PATH}/../.. -name "*.gcno" | xargs  -i mv {} ${FUZZ_GENERATE_DIR}

    lcov --d ${FUZZ_GENERATE_DIR} --c --output-file ${FUZZ_GENERATE_DIR}/cover.info --rc lcov_branch_coverage=1 --rc lcov_excl_line=ACCDATA_*
    if [ 0 != $? ];then
      echo "Failed to generate all coverage info"
      exit 1
    fi

    gcc_version=$(gcc -dumpversion 2>/dev/null)
    if [ -z "$gcc_version" ]; then
        echo "ERROR: GCC not found or version unavailable" >&2
        exit 1
    fi

    lcov -r ${FUZZ_GENERATE_DIR}/cover.info "*${gcc_version}*" -o ${FUZZ_GENERATE_DIR}/cover.info --rc lcov_branch_coverage=1 --rc lcov_excl_line=ACCDATA_*
    if [ 0 != $? ];then
      echo "Failed to remove *${gcc_version}* from coverage info"
      exit 1
    fi

    lcov -r ${FUZZ_GENERATE_DIR}/cover.info "*test*" -o ${FUZZ_GENERATE_DIR}/cover.info --rc lcov_branch_coverage=1 --rc lcov_excl_line=ACCDATA_*
    if [ 0 != $? ];then
      echo "Failed to remove *tests/ut* from coverage info"
      exit 1
    fi

    lcov -r ${FUZZ_GENERATE_DIR}/cover.info "*output*" -o ${FUZZ_GENERATE_DIR}/cover.info --rc lcov_branch_coverage=1 --rc lcov_excl_line=ACCDATA_*
    if [ 0 != $? ];then
      echo "Failed to remove *install* from coverage info"
      exit 1
    fi

    genhtml -o ${FUZZ_GENERATE_DIR}/result ${FUZZ_GENERATE_DIR}/cover.info --show-details --legend --rc lcov_branch_coverage=1 --rc lcov_excl_line=ACCDATA_*
    if [ 0 != $? ];then
      echo "Failed to generate all coverage info with html format"
      exit 1
    fi
}

dt_fuzz_report
# **************************报告生成 End ***************************