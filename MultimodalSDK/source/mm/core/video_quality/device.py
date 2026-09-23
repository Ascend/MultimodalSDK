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

"""Device resolver — maps ``"auto"`` to the best available device.

Resolution order: **NPU > CPU** (GPU is not supported, consistent with the
SDK's existing device scope).
"""

from __future__ import annotations


_VALID_DEVICES = {"auto", "npu", "cpu"}


def _npu_available() -> bool:
    """Return True when torch_npu is importable and an NPU device is visible."""
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401

        return torch.npu.is_available() and torch.npu.device_count() > 0
    except Exception:
        return False


class DeviceResolver:
    """Resolve a device string to a concrete ``torch.device``-compatible string."""

    @staticmethod
    def resolve(device: str = "auto") -> str:
        """Resolve *device* to one of ``"npu"`` / ``"cpu"``.

        Args:
            device: One of ``"auto"``, ``"npu"``, ``"cpu"``.

        Returns:
            A concrete device string (``"npu"`` if NPU is available,
            otherwise ``"cpu"``).

        Raises:
            ValueError: If *device* is not a recognised value or is ``"cuda"``.
        """
        if device not in _VALID_DEVICES:
            if device == "cuda":
                raise ValueError("GPU (cuda) is not supported. Use 'auto', 'npu', or 'cpu'.")
            raise ValueError(f"Unsupported device: {device!r}. Supported: {sorted(_VALID_DEVICES)}")

        if device == "auto":
            return "npu" if _npu_available() else "cpu"
        return device

    @staticmethod
    def resolve_torch_device(device: str = "auto"):
        """Return a ``torch.device`` object for *device*."""
        import torch

        resolved_device = DeviceResolver.resolve(device)
        return torch.device(resolved_device)

    @staticmethod
    def default_frame_batch_size(device: str = "auto") -> int:
        """Return the default frame batch size for *device*.

        NPU: 2 (memory constrained), CPU: 1.
        """
        resolved_device = DeviceResolver.resolve(device)
        return 2 if resolved_device.startswith("npu") else 1
