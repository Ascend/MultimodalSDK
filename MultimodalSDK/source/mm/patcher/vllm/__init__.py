#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
"""
vLLM plugin entry point.

The ``patch()`` function below is the side-effecting entry point that
vLLM's plugin discovery mechanism imports at process startup. It uses the
``MM_*`` environment variables (declared in :mod:`.constants`) to decide
which individual monkey-patch modules to load:

* ``MM_SCC_RATE < 1.0``     -> enable SCC token compression patches
* ``MM_PREPROCESSOR``       -> enable MultimodalSDK preprocessor patches

This file is original to MultimodalSDK and is not adapted from upstream.
"""

from .constants import MM_PREPROCESSOR, MM_SCC_RATE
from ...comm.log import _Logger as _Log


def patch():
    """Register SDK patches onto vLLM classes based on env vars."""

    if MM_SCC_RATE < 1:
        from . import patch_qwen2_5_vl  # noqa: F401
        from . import patch_qwen3_vl  # noqa: F401

        _Log.info(f"patch scc rate={MM_SCC_RATE}")

    if MM_PREPROCESSOR:
        from . import patch_qwen2_5_processor  # noqa: F401
        from . import patch_qwen3_processor  # noqa: F401

        _Log.info("patch MultimodalSDK preprocessor")
