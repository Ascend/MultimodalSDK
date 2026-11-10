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

from importlib.metadata import PackageNotFoundError, version

AVAILABLE = False
__all__ = []

try:
    _transformers_major = int(version("transformers").split(".")[0])
except (PackageNotFoundError, ValueError):
    _transformers_major = -1

# Legacy SDK adapters use Transformers 4.x APIs. FlashVid uses vLLM's
# processors and must also load when the newer image uses Transformers 5.x.
if 0 <= _transformers_major < 5:
    from .qwen2_vl_preprocessor import MultimodalQwen2VLImageProcessor
    from .internvl2_preprocessor import InternVL2PreProcessor

    AVAILABLE = True
    __all__ = ['MultimodalQwen2VLImageProcessor', 'InternVL2PreProcessor']
