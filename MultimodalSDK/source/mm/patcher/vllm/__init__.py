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

"""Optional FlashVid model patches for vLLM."""

from importlib import import_module

from mm.comm.log import _Logger as logger
from mm.patcher.vllm.constants import ENABLE_FLASHVID


def patch_mm():
    """Register FlashVid model patches when enabled."""
    if ENABLE_FLASHVID:
        for module_name in ("patch_qwen2_5_vl", "patch_qwen3_vl", "patch_qwen3_5"):
            import_module(f"{__name__}.flashvid.{module_name}")

        logger.info("patch flash vid enabled")
