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
MERGE_BUILD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

ACC_SDK_ROOT_DIR="${MERGE_BUILD_DIR}/AccSDK"
ACC_BUILD_DIR="${ACC_SDK_ROOT_DIR}/build_script"
cd "$ACC_BUILD_DIR" || { echo "Error: Cannot enter directory $ACC_BUILD_DIR"; exit 1; }

chmod +x build.sh
if [[ "$1" == "test" ]]; then
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/opensource/FFmpeg/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/opensource/libjpeg-turbo/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/output/lib:${LD_LIBRARY_PATH}"
    ./build.sh test || exit 1
    export GTEST_HOME="${ACC_SDK_ROOT_DIR}/acc_data/3rdparty/gtest/googletest/build/googletest"
    export LD_LIBRARY_PATH="${ACC_SDK_ROOT_DIR}/acc_data/3rdparty/gtest/googletest/build/lib/:${LD_LIBRARY_PATH}"
    cd ../build && make test
    cat ./Testing/Temporary/LastTest.log && cd ../build_script && bash gen_report.sh  && python3 testcases_xml_report.py ../test coverage-report
else
    ./build.sh || exit 1
fi

MULTI_SDK_ROOT_DIR="${MERGE_BUILD_DIR}/MultimodalSDK"
MULTI_BUILD_DIR="${MULTI_SDK_ROOT_DIR}/build_script"
cd "$MULTI_BUILD_DIR" || { echo "Error: Cannot enter directory $MULTI_BUILD_DIR"; exit 1; }

export PYTHONPATH=${MULTI_SDK_ROOT_DIR}/source/:${PYTHONPATH}
export LD_LIBRARY_PATH=/opt/python3.11.4/lib/python3.11/site-packages/torch/lib/:$LD_LIBRARY_PATH

chmod +x build.sh
if [[ "$1" == "test" ]]; then
    ./build.sh test || exit 1
else
    ./build.sh || exit 1
fi

PACKAGE_DIR="${MERGE_BUILD_DIR}/makeself"
cd "$PACKAGE_DIR" || { echo "Error: Cannot enter directory $PACKAGE_DIR"; exit 1;}

patch -p1 < ../makeself_patch/makeself-2.5.0.patch || { echo "Error: Patch failed"; exit 1;}

cd "$SCRIPT_DIR" || { echo "Error: Cannot enter directory $SCRIPT_DIR"; exit 1;}
chmod +x package.sh
./package.sh || { echo "Error: Packaging failed"; exit 1;}

echo "Packaging completed successfully!"