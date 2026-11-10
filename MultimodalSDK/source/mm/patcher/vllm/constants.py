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
"""FlashVid environment configuration, read once when the plugin loads."""

import os

from mm.comm.log import _Logger as _Log


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


ENABLE_FLASHVID = _parse_bool(
    "ENABLE_FLASHVID",
    os.environ.get("ENABLE_FLASHVID", ""),
    default=False,
)
