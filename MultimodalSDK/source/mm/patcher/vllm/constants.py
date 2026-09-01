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
SDK-controlled environment variables for the vLLM patches.

All values are read once at import time. vLLM loads this module during
its plugin discovery (via ``mm.patcher.vllm.patch``), so any change to
these env vars after vLLM startup will not take effect.

This file is original to MultimodalSDK and is not adapted from upstream.
"""

import os

from ...comm.log import _Logger as _Log


def _clamp_float(name: str, raw: str, default: float, lo: float, hi: float, *, min_exclusive: bool = False) -> float:
    """Parse an env var as float, validate against ``[lo, hi]`` or ``(lo, hi]``.

    By default the lower bound is inclusive (``lo <= value <= hi``). Pass
    ``min_exclusive=True`` to require ``lo < value <= hi``; in that mode
    a value at or below ``lo`` falls back to ``default`` (it cannot be
    clamped to a still-valid value).
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _Log.warn(f"{name}={raw!r} is not a valid float; falling back to default={default}")
        return default
    if min_exclusive and value <= lo:
        _Log.warn(f"{name}={value} must be > {lo}; falling back to default={default}")
        return default
    if value < lo or value > hi:
        clamped = max(lo, min(hi, value))
        _Log.warn(f"{name}={value} is out of range [{lo}, {hi}]; clamped to {clamped}")
        return clamped
    return value


def _clamp_int(name: str, raw: str, default: int, lo: int, hi: int) -> int:
    """Parse an env var as int, clamp to ``[lo, hi]`` and warn if needed."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        _Log.warn(f"{name}={raw!r} is not a valid integer; falling back to default={default}")
        return default
    if value < lo or value > hi:
        clamped = max(lo, min(hi, value))
        _Log.warn(f"{name}={value} is out of range [{lo}, {hi}]; clamped to {clamped}")
        return clamped
    return value


def _parse_bool(name: str, raw: str, default: bool) -> bool:
    """Parse an env var as bool (true/false/1/0/yes/no/on/off)."""
    if not raw:
        return default
    low = raw.strip().lower()
    if low in ("1", "true", "yes", "on", "y", "t"):
        return True
    if low in ("0", "false", "no", "off", "n", "f"):
        return False
    _Log.warn(f"{name}={raw!r} is not a valid bool; falling back to default={default}")
    return default


# MM_SCC_RATE: float in (0, 1]. 1.0 disables compression.
MM_SCC_RATE = _clamp_float(
    "MM_SCC_RATE", os.environ.get("MM_SCC_RATE", "1"), default=1.0, lo=0.0, hi=1.0, min_exclusive=True
)

# MM_SCC_TAU: float in (0, 1]. Cosine-similarity threshold for clustering.
MM_SCC_TAU = _clamp_float(
    "MM_SCC_TAU", os.environ.get("MM_SCC_TAU", "0.95"), default=0.95, lo=0.0, hi=1.0, min_exclusive=True
)

# MM_SCC_EPSILON: float in (0, 1). Approx Union-Find sampling error tolerance.
MM_SCC_EPSILON = _clamp_float(
    "MM_SCC_EPSILON", os.environ.get("MM_SCC_EPSILON", "0.05"), default=0.05, lo=0.0, hi=1.0, min_exclusive=True
)

# MM_SCC_MAX_TOKENS_PER_ITEM: int in [0, 65536]. 0 = no limit.
MM_SCC_MAX_TOKENS_PER_ITEM = _clamp_int(
    "MM_SCC_MAX_TOKENS_PER_ITEM",
    os.environ.get("MM_SCC_MAX_TOKENS_PER_ITEM", "8192"),
    default=8192,
    lo=0,
    hi=65536,
)

# MM_PREPROCESSOR: bool. Empty / unset -> False (use SDK preprocessor off by default).
MM_PREPROCESSOR = _parse_bool(
    "MM_PREPROCESSOR",
    os.environ.get("MM_PREPROCESSOR", ""),
    default=False,
)
