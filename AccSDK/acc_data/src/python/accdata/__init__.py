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

import os
import ctypes


current_dir = os.path.dirname(os.path.abspath(__file__))
lib_file = "lib_accdata.so"
lib_path = os.path.join(current_dir, lib_file)
if not os.path.exists(lib_path):
    raise FileNotFoundError(
        f"Shared library not found at: {lib_path}\n"
        f"Current directory contents: {os.listdir(package_dir)}"
    )
try:
    lib = ctypes.CDLL(lib_path)
except Exception as e:
    raise RuntimeError(f"Failed to load library {lib_path}: {str(e)}") from e


from accdata.pipeline import Pipeline
from accdata.ops import external_source, to_tensor, resize_crop, normalize, to_tensor_resize_crop_norm, qwen_fusion_op
from accdata.plugin.pytorch import to_torch_tensorlist, to_accdata_tensorlist


__all__ = [
    'Pipeline',
    'external_source',
    'to_tensor',
    'resize_crop',
    'normalize',
    'to_tensor_resize_crop_norm',
    'qwen_fusion_op',
    'to_torch_tensorlist',
    'to_accdata_tensorlist',
]