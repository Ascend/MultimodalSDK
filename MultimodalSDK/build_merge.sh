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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Start building AccSDK..."

ACC_BUILD_DIR="${SCRIPT_DIR}/AccSDK/build_script"
cd "$ACC_BUILD_DIR" || { echo "Error: Cannot enter directory $ACC_BUILD_DIR"; exit 1; }

chmod +x build.sh
if [[ "$1" == "test" ]]; then
    ./build.sh test || exit 1
    cd ../build && make test
    cat ./Testing/Temporary/LastTest.log && cd ../build_script && bash gen_report.sh  && python3 testcases_xml_report.py ../test coverage-report
else
    ./build.sh || exit 1
fi

echo "AccSDK build completed, starting MultimodalSDK..."

MULTI_BUILD_DIR="${SCRIPT_DIR}/MultimodalSDK/build_script"
cd "$MULTI_BUILD_DIR" || { echo "Error: Cannot enter directory $MULTI_BUILD_DIR"; exit 1; }

chmod +x build.sh
if [[ "$1" == "test" ]]; then
    ./build.sh test || exit 1
    cd ../build && make test
    cat ./Testing/Temporary/LastTest.log && cd ../build_script && bash gen_report.sh  && python3 testcases_xml_report.py ../test coverage-report
else
    ./build.sh || exit 1
fi

echo "All build tasks completed!"