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
# Path validation utilities for file and directory existence checks.

from pathlib import Path
from typing import Union


def validate_path_exists(path: Union[str, Path], label: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p.resolve().as_posix()}" + (f" ({label})" if label else ""))
    return p


def validate_dir_exists(path: Union[str, Path], label: str = "") -> Path:
    p = validate_path_exists(path, label)
    if not p.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {p.resolve().as_posix()}" + (f" ({label})" if label else "")
        )
    return p


def validate_file_exists(path: Union[str, Path], label: str = "") -> Path:
    p = validate_path_exists(path, label)
    if not p.is_file():
        raise IsADirectoryError(f"Path is not a file: {p.resolve().as_posix()}" + (f" ({label})" if label else ""))
    return p
