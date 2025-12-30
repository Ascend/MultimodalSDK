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

if [ "$1" = "-h" -o "$1" = "--help" ]; then
    echo "Example:"
    echo "    $0 -h                                           Show this message"
    echo "    $0 --run_mode dt_mode                           Specify dt_mode, default is smoke, can be [all, to_tensor, benchmark, qwen]"
    exit
fi

run_mode="smoke"
while [ $# -ge 1 ]
do
    if [ "$1" = "--run_mode" ]; then
        shift
        run_mode="$1"
        shift
    else
        break
    fi
done

python_env_ok=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd)"
TOP_DIR="${SCRIPT_DIR}/../.."
SECUREC_DIR="usr/local/Ascend/driver"
PYTHON_UT_DIR=${TOP_DIR}/test/python
SRC_DIR=${TOP_DIR}/src
LIB_HW_SECURE=${SECUREC_DIR}/lib64/common

function check_whl_not_installed() {
    PACKAGE_NAME="$1"

    # 检查包是否存在
    if pip3 list --format=columns | grep -E "^$PACKAGE_NAME\s+" &> /dev/null; then
        echo "$PACKAGE_NAME installed, skip."
    else
        python_env_ok=0
    fi
}

function python_dt_env_prepare() {
    package_names=("pytest" "pytest-cov" "lxml" "hw-hdt-cli" "transformers")

    for pkg in "${package_names[@]}"; do
        check_whl_not_installed $pkg
    done

    if [ $python_env_ok -eq 0 ]; then
        pip3 install pytest \
        -i https://cmc.centralrepo.rnd.huawei.com/artifactory/pypi-central-repo/simple \
        --extra-index-url https://pypi.cloudartifact.dgg.dragon.tools.huawei.com/artifactory/api/pypi/pypi-oss/simple/ \
        --trusted-host cmc.centralrepo.rnd.huawei.com

        pip3 install pytest-cov \
        -i https://cmc.centralrepo.rnd.huawei.com/artifactory/pypi-central-repo/simple \
        --extra-index-url https://pypi.cloudartifact.dgg.dragon.tools.huawei.com/artifactory/api/pypi/pypi-oss/simple/ \
        --trusted-host cmc.centralrepo.rnd.huawei.com

        pip3 install lxml \
        -i https://cmc.centralrepo.rnd.huawei.com/artifactory/product_pypi/simple \
        --extra-index-url https://cmc.centralrepo.rnd.huawei.com/pypi/simple \
        --trusted-host cmc.centralrepo.rnd.huawei.com

        pip3 install hw-hdt-cli \
        -i https://cmc.centralrepo.rnd.huawei.com/artifactory/product_pypi/simple \
        --extra-index-url https://cmc.centralrepo.rnd.huawei.com/artifactory/pypi-central-repo/simple \
        --trusted-host cmc.centralrepo.rnd.huawei.com

        pip3 install transformers \
        -i https://cmc.centralrepo.rnd.huawei.com/artifactory/product_pypi/simple \
        --extra-index-url https://cmc.centralrepo.rnd.huawei.com/artifactory/pypi-central-repo/simple \
        --trusted-host cmc.centralrepo.rnd.huawei.com
    fi
}

function check_install_pytest_benchmark() {
    if check_whl_not_installed pytest-benchmark; then
        echo "pytest-benchmark installed, skip."
    else
        pip3 install pytest-benchmark \
        -i https://cmc.centralrepo.rnd.huawei.com/artifactory/pypi-central-repo/simple \
        --extra-index-url https://pypi.cloudartifact.dgg.dragon.tools.huawei.com/artifactory/api/pypi/pypi-oss/simple/ \
        --trusted-host cmc.centralrepo.rnd.huawei.com
    fi
}

python_dt_env_prepare

cp ${TOP_DIR}/build/lib.linux-aarch64-*/accdata/backend.cpython-* ${SRC_DIR}/python/accdata
cp ${TOP_DIR}/build/lib.linux-aarch64-*/accdata/lib_accdata.so ${SRC_DIR}/python/accdata

cd ${PYTHON_UT_DIR}

# RUN python dt
export PYTHONPATH=$PYTHONPATH:${PYTHON_UT_DIR}
export PYTHONPATH=$PYTHONPATH:${SRC_DIR}/python
export LD_LIBRARY_PATH=$LIB_HW_SECURE:$LD_LIBRARY_PATH

hdt -V
if [ "${run_mode}" == "all" ]; then
    hdt test -c on --args="--junitxml=final.xml --durations=0 -m \"smoke or slow\""
elif [ "${run_mode}" == "to_tensor" ]; then
    hdt test -c on --args="--junitxml=final.xml --durations=0 -m \"to_tensor\""
elif [ "${run_mode}" == "qwen" ]; then
    hdt test -c on --args="--junitxml=final.xml --durations=0 -m \"qwen\""
elif [ "${run_mode}" == "benchmark" ]; then
    check_install_pytest_benchmark
    taskset -c 0-8 pytest benchmark --benchmark-sort=NAME
else
    hdt test -c on --args="--junitxml=final.xml --durations=0 -m \"smoke\""
fi
