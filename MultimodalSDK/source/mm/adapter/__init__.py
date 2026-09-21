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
# adapter仅适配vLLM 4以前的版本（对应transformers 4.x）：依赖transformers的
# Qwen2VLImageProcessor / BatchFeature等旧接口。transformers >= 5（新构建环境，
# 见build_script/build.sh安装transformers==5.5.4）或未安装transformers时不再引入，
# 避免import mm报错。
from importlib.metadata import version as _pkg_version, PackageNotFoundError

AVAILABLE = False

try:
    _transformers_major = int(_pkg_version("transformers").split(".")[0])
except (PackageNotFoundError, ValueError):
    _transformers_major = -1

if 0 <= _transformers_major < 5:
    from .qwen2_vl_preprocessor import MultimodalQwen2VLImageProcessor  # noqa: F401
    from .internvl2_preprocessor import InternVL2PreProcessor  # noqa: F401

    AVAILABLE = True
