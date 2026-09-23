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

"""Root conftest — mocks native lib dependencies so tests can run without libcore.so.

This file is loaded by pytest before any test module imports.  It checks whether
``mm.acc._impl`` can be imported normally; if not (libcore.so missing), it
installs a mock ``acc`` module so that ``import mm`` succeeds.
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock


def _ensure_mm_importable():
    """Ensure ``import mm`` works even without the native C++ library."""
    try:
        import mm  # noqa: F401

        return
    except ImportError:
        pass

    # Create mock modules for the native acc layer
    mock_acc = MagicMock()
    mock_acc_impl = MagicMock()
    mock_wrapper = MagicMock()

    # Register mocks in sys.modules BEFORE importing mm
    sys.modules["mm.acc"] = mock_acc
    sys.modules["mm.acc._impl"] = mock_acc_impl
    sys.modules["mm.acc._impl.acc"] = MagicMock()
    sys.modules["mm.acc.wrapper"] = mock_wrapper
    for sub in ["tensor_wrapper", "image_wrapper", "video_wrapper", "audio_wrapper", "data_type", "util"]:
        sys.modules[f"mm.acc.wrapper.{sub}"] = MagicMock()

    # Now mm should be importable
    try:
        import mm  # noqa: F401
    except Exception as exc:
        warnings.warn(
            f"import mm failed even with mocked native modules: {exc!r}. "
            "Tests importing mm will fail during collection.",
            stacklevel=2,
        )


_ensure_mm_importable()
