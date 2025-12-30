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
# Description: SDK package script.
# Author: Multimodal SDK
# Create: 2025
# History: NA
set -e

echo "==============Start packing=============="

# MultimodalSDK目录
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly SDK_VERSION=$(sed -n 's/^version:[[:space:]]*//p' "${ROOT_DIR}"/../ci/config/config.ini)
cp -f ${ROOT_DIR}/MultimodalSDK/output/script/install.sh ${ROOT_DIR}/MultimodalSDK/output/
# 创建输出目录
OUTPUT_DIR="${ROOT_DIR}/output/"
mkdir -p "${OUTPUT_DIR}"

echo "Starting makeself packaging..."
bash ${ROOT_DIR}/makeself/makeself.sh --chown --nomd5 --sha256 --nocrc \
    --header ${ROOT_DIR}/makeself/makeself-header.sh \
    --help-header ${ROOT_DIR}/MultimodalSDK/output/script/help.info \
    --packaging-date "" \
    --tar-extra '--owner=root --group=root' \
    ${ROOT_DIR}/MultimodalSDK/output/ ${OUTPUT_DIR}/Ascend-mindxsdk-multimodal_${SDK_VERSION}_linux-aarch64.run "ASCEND Multimodal SDK RUN PACKAGE" ./install.sh

echo "==============Packaging completed successfully=============="
echo "Output file: Ascend-mindxsdk-multimodal_${SDK_VERSION}_linux-aarch64.run"
