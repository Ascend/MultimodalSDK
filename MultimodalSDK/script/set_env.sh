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
# Description: Multimodal SDK env set script
# Author: Multimodal SDK
# Create: 2025
# History: NA

export MULTIMODAL_SDK_HOME=""

export LD_LIBRARY_PATH="${MULTIMODAL_SDK_HOME}/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${MULTIMODAL_SDK_HOME}/opensource/FFmpeg/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${MULTIMODAL_SDK_HOME}/opensource/libjpeg-turbo/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${MULTIMODAL_SDK_HOME}/opensource/soxr/lib:$LD_LIBRARY_PATH"
