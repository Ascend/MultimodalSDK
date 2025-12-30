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
from enum import Enum


class DataType(Enum):
    INT8 = 2
    UINT8 = 4
    FLOAT32 = 0


class TensorFormat(Enum):
    ND = 2
    NHWC = 1
    NCHW = 0


class ImageFormat(Enum):
    RGB = 12
    BGR = 13
    RGB_PLANAR = 69
    BGR_PLANAR = 70


class Interpolation(Enum):
    BICUBIC = 2


class DeviceMode(Enum):
    CPU = 0
