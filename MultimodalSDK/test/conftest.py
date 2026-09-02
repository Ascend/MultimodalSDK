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
"""Pytest bootstrap for the mm package.

On a target environment (Linux with the built ``_acc`` extension) the real
``mm`` package is imported as-is. On a development machine without the SWIG
build artifacts, importing ``mm`` fails at ``libcore.so`` preloading; in that
case minimal stub modules for ``mm`` / ``mm.core`` / ``mm.comm`` /
``mm.comm.log`` and the ``mm.acc`` wrapper chain (pulled in by
``mm.core.segmenter`` package imports) are injected so pure-Python modules
(e.g. ``mm.core.segmenter``) stay testable. Tests that require the real
``_acc`` fail on import either way, exactly as before.
"""

import sys
import types
from pathlib import Path

try:
    import mm  # noqa: F401
except Exception:  # noqa: BLE001  no usable real mm package here
    _mm_root = Path(__file__).resolve().parents[1] / "source" / "mm"

    _mm_pkg = types.ModuleType("mm")
    _mm_pkg.__path__ = [str(_mm_root)]
    sys.modules.setdefault("mm", _mm_pkg)

    _core_pkg = types.ModuleType("mm.core")
    _core_pkg.__path__ = [str(_mm_root / "core")]
    sys.modules.setdefault("mm.core", _core_pkg)

    _comm_pkg = types.ModuleType("mm.comm")
    _comm_pkg.__path__ = [str(_mm_root / "comm")]
    sys.modules.setdefault("mm.comm", _comm_pkg)

    _log_mod = types.ModuleType("mm.comm.log")

    class _StubLogger:
        @staticmethod
        def debug(message):  # noqa: D102
            pass

        @staticmethod
        def info(message):  # noqa: D102
            pass

        @staticmethod
        def warn(message):  # noqa: D102
            pass

        @staticmethod
        def error(message):  # noqa: D102
            pass

        @staticmethod
        def fatal(message):  # noqa: D102
            pass

    _log_mod._Logger = _StubLogger
    sys.modules.setdefault("mm.comm.log", _log_mod)

    # SWIG-based acc wrapper chain: stub video_decode / video_info so that
    # ``mm.core.segmenter`` package imports resolve on any machine.
    _acc_pkg = types.ModuleType("mm.acc")
    sys.modules.setdefault("mm.acc", _acc_pkg)

    _acc_wrapper_pkg = types.ModuleType("mm.acc.wrapper")
    sys.modules.setdefault("mm.acc.wrapper", _acc_wrapper_pkg)

    if "mm.acc.wrapper.video_wrapper" not in sys.modules:
        _video_wrapper = types.ModuleType("mm.acc.wrapper.video_wrapper")
        _video_wrapper.video_decode = lambda *a, **k: []
        _video_wrapper.video_info = lambda *a, **k: {}
        sys.modules["mm.acc.wrapper.video_wrapper"] = _video_wrapper
