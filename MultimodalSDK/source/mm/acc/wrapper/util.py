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


def _ensure_bytes(value, name: str) -> bytes:
    """
    Convert a str to bytes (UTF-8) or ensure it's bytes.
    Reject any bytes containing NULL (\x00) to prevent C++ strcmp bypass.
    """
    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        raise TypeError(f"{name} must be str or bytes")
    if b'\x00' in value:
        raise ValueError(f"{name} contains NULL byte (\\x00), which is not allowed")
    return value


class ObjectWrapper:
    """
    Temporary wrapper class to implement numpy's __array_interface__ protocol

    numpy's asarray() function checks if an object has the __array_interface__ attribute.
    If it exists, NumPy uses this interface to create an ndarray without copying data.
    This enables zero-copy data conversion.

    Args:
        __array_interface__ (dict): __array_interface__ protocol dictionary
    """

    def __init__(self, interface_dict):
        # Set the NumPy array interface containing data pointer, shape, dtype, etc.
        self.__array_interface__ = interface_dict.get('__array_interface__')