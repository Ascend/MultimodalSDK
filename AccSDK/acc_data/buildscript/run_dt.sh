#!/bin/bash
# Perform  build inference
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

# print usage message
function usage() {
    echo "Accdata UT"
    echo "USAGE:"
    echo "bash $0 [-b true | false] [-s] [-a] [-t UT_TYPE][-h]"
    echo "e.g. $0 -b, default run smoke cases"
    echo ""
    echo "OPTIONS:"
    echo "  -b Build src, default true, or false"
    echo "  -s Run smoke cases (default case)"
    echo "  -a Run all cases"
    echo "  -t Run ut type: default all, or cpp, py"
    echo "  -h Print usage"
}

function checkopts() {
    run_mode="smoke"
    build_src="true"
    ut_type="all"

    # Process the options
    while getopts 't:b:sah' opt
    do
        case "${opt}" in
        b)  build_src_array=("true" "false")
            echo "${build_src_array[@]}" | grep -wq "$OPTARG" &&  right_key=1 || right_key=0
             if [ "${right_key}" == 1 ]; then
                build_src="$OPTARG"
             else
                echo "Unrecognized src_build param, default is true, or false , ignored"
             fi
            ;;
        s)  run_mode="smoke"
            ;;
        a)  run_mode="all"
            ;;
        t)  ut_type_array=("cpp" "py" "all")
            echo "${ut_type_array[@]}" | grep -wq "$OPTARG" &&  right_key=1 || right_key=0
            if [ "${right_key}" == 1 ]; then
                ut_type="$OPTARG"
            else
                echo "Unrecognized ut_type param, default is py, or cpp / all, ignored"
            fi
            ;;
        h)  usage
            exit 0
            ;;
        ?)  echo "Unknown option ${opt}!"
            usage
            exit 1
        esac
    done
}

# check options
checkopts $@

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd)"
TOP_DIR="${SCRIPT_DIR}/.."

echo "[Accdata] Run UT Info: build_src: ${build_src}, run_mode: ${run_mode}, ut_type: ${ut_type}"

if [ "${build_src}" == "false" ]; then
    if [ "${ut_type}" == "all" ]; then
        echo "If ut_type is all, build_src should be true!!"
        exit 1
    fi
fi

if [ "$ut_type" == "cpp" ] || [ "$ut_type" == "all" ]; then
    if [ "${build_src}" == "true" ]; then
        echo "src build for cpp ut!"
        bash ${TOP_DIR}/build.sh -t ut
    else
        echo "src build disabled, please make sure src is built before running!"
    fi

    if [ "$run_mode" == "smoke" ]; then
        bash ${TOP_DIR}/test/cpp/run_dt.sh --disable_src_build
    else
        bash ${TOP_DIR}/test/cpp/run_dt.sh --disable_src_build
    fi

fi

if [ "$ut_type" == "py" ] || [ "$ut_type" == "all" ]; then
    if [ "${build_src}" == "true" ]; then
        echo "src build for py ut!"
        bash ${TOP_DIR}/build.sh
    else
        echo "src build disabled, please make sure src is built before running!"
    fi
    bash ${TOP_DIR}/test/python/run_dt.sh $run_mode
fi
