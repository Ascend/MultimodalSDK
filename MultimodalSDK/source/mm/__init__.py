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
from .acc import (Tensor, TensorFormat, DataType, Image, ImageFormat, DeviceMode, Interpolation, video_decode,
                  normalize, load_audio)
from .comm import LogLevel, register_log_conf
from .adapter import MultimodalQwen2VLImageProcessor, InternVL2PreProcessor

__all__ = ["Tensor", 'DataType', 'TensorFormat', 'Image', 'ImageFormat', 'LogLevel', 'register_log_conf', 'DeviceMode',
           'Interpolation', 'video_decode', 'normalize', 'MultimodalQwen2VLImageProcessor', 'InternVL2PreProcessor',
           'load_audio']
register_log_conf(LogLevel.INFO, None)
