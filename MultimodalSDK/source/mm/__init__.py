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
from .acc import (
    Tensor,
    TensorFormat,
    DataType,
    Image,
    ImageFormat,
    DeviceMode,
    Interpolation,
    video_decode,
    normalize,
    load_audio,
)
from .comm import LogLevel, register_log_conf
from .core import BaseFrameSelector, KFrameSelector, KRangFrameSelector
from . import adapter

__all__ = [
    'Tensor',
    'DataType',
    'TensorFormat',
    'Image',
    'ImageFormat',
    'LogLevel',
    'register_log_conf',
    'DeviceMode',
    'Interpolation',
    'video_decode',
    'normalize',
    'load_audio',
    'BaseFrameSelector',
    'KFrameSelector',
    'KRangFrameSelector',
]

# adapter仅在transformers 4.x（对应vLLM 4以前的版本）可用，见 mm/adapter/__init__.py；
# 不可用时这两个类不导出，mm其余功能不受影响。
if adapter.AVAILABLE:
    from .adapter import MultimodalQwen2VLImageProcessor, InternVL2PreProcessor

    __all__ += ['MultimodalQwen2VLImageProcessor', 'InternVL2PreProcessor']
del adapter

register_log_conf(LogLevel.INFO, None)
