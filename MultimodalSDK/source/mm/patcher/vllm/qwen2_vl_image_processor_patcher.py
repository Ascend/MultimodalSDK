#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import transformers
from ...adapter.qwen2_vl_preprocessor import MultimodalQwen2VLImageProcessor
from ...comm.log import _Logger as log

# Save the original function for fallback
raw_func = transformers.models.auto.image_processing_auto.get_image_processor_class_from_name


def get_image_processor_class_from_name(class_name: str):
    """
    Patch the Transformers image processor class retrieval to support
    MultimodalQwen2VLImageProcessor.

    - Only compatible with Transformers versions 4.48.0 to 4.51.3.
    - For class_name == "Qwen2VLImageProcessor", return the patched processor.
    - Otherwise, fallback to the original function.
    """
    if class_name == "Qwen2VLImageProcessor":
        log.info("Multimodal SDK Qwen2 VL Image Patcher Enabled!")
        return MultimodalQwen2VLImageProcessor
    raw_processor = raw_func(class_name)
    return raw_processor

# Override the Transformers function with the patched version
transformers.models.auto.image_processing_auto.get_image_processor_class_from_name = get_image_processor_class_from_name
