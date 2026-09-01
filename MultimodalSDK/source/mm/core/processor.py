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
Image and video preprocessing utilities.

This module provides a unified resize-and-normalize interface used by
Qwen2-VL / Qwen3-VL processors. The actual NPU-accelerated preprocessing
is delegated to ``acc.Qwen2VLProcessor.PreprocessTensor`` from the
underlying acc package.
"""

from typing import List

import numpy as np
import torch

from ..comm.log import _Logger as _Log
from ..acc._impl import acc  # pylint: disable=no-name-in-module
from ..acc.wrapper.util import ObjectWrapper


def resize_and_normalize(
    frames: torch.Tensor, height: int, width: int, image_mean: List[float], image_std: List[float]
) -> torch.Tensor:
    """Resize and normalize a batch of image/video frames on the NPU.

    The function accepts a 4D tensor in either NCHW (``(N, 3, H, W)``) or
    NHWC (``(N, H, W, 3)``) layout, validates the inputs, dispatches the
    tensor to the underlying ``acc.Qwen2VLProcessor.PreprocessTensor``
    operator and returns the result wrapped as a ``torch.Tensor``.

    Args:
        frames: Input tensor with shape ``(N, C, H, W)`` or ``(N, H, W, C)``.
        height: Target resize height in pixels. Must be > 0.
        width: Target resize width in pixels. Must be > 0.
        image_mean: Per-channel normalization mean. Must be a list/tuple of
            3 floats in ``[0.0, 1.0]``. Out-of-range values raise
            ``ValueError``.
        image_std: Per-channel normalization std. Must be a list/tuple of
            3 positive floats.

    Returns:
        A ``torch.Tensor`` containing the resized and normalized frames.

    Raises:
        ValueError: If any input fails the validation checks (shape, sign,
            length, mean range or std positivity).
    """
    if len(frames.shape) != 4:
        error_msg = f"frames shape is not permitted, expected 4D tensor but got shape {frames.shape}"
        _Log.error(error_msg)
        raise ValueError(error_msg)

    if frames.dtype != torch.uint8:
        error_msg = f"frames dtype is not permitted, expected torch.uint8 but got {frames.dtype}"
        _Log.error(error_msg)
        raise ValueError(error_msg)

    if height <= 0 or width <= 0:
        error_msg = f"height and width must be positive, got height={height}, width={width}"
        _Log.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(image_mean, (list, tuple)) or len(image_mean) != 3:
        error_msg = f"image_mean must be a list/tuple of 3 elements, got {image_mean}"
        _Log.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(image_std, (list, tuple)) or len(image_std) != 3:
        error_msg = f"image_std must be a list/tuple of 3 elements, got {image_std}"
        _Log.error(error_msg)
        raise ValueError(error_msg)

    for i, val in enumerate(image_mean):
        if not (0.0 <= val <= 1.0):
            error_msg = f"image_mean[{i}]={val} is out of range [0.0, 1.0], expected normalized value"
            _Log.error(error_msg)
            raise ValueError(error_msg)

    for i, val in enumerate(image_std):
        if val <= 0.0:
            error_msg = f"image_std[{i}]={val} must be positive"
            _Log.error(error_msg)
            raise ValueError(error_msg)

    _Log.debug(f"ResizeAndNormalize: input shape={frames.shape}, target size=({height}, {width})")

    # Determine tensor format: NCHW if dim 1 is channel (3), otherwise NHWC if dim 3 is channel (3)
    is_nchw = frames.shape[1] == 3
    is_nhwc = frames.shape[3] == 3

    if not is_nchw and not is_nhwc:
        error_msg = f"frames shape {frames.shape} is not NCHW or NHWC, neither dim 1 nor dim 3 equals 3"
        _Log.error(error_msg)
        raise ValueError(error_msg)

    if is_nchw:
        _Log.debug("Detected NCHW format")
        contiguous_frames = np.ascontiguousarray(frames.numpy())
        acc_tensor = acc.Tensor.from_numpy(contiguous_frames)
        acc_tensor.set_format(acc.TensorFormat_NCHW)
    else:
        _Log.debug("Detected NHWC format")
        contiguous_frames = np.ascontiguousarray(frames.numpy())
        acc_tensor = acc.Tensor.from_numpy(contiguous_frames)
        acc_tensor.set_format(acc.TensorFormat_NHWC)

    tensor_acc_list = acc.Qwen2VLProcessor.PreprocessTensor([acc_tensor], image_mean, image_std, width, height)
    return torch.tensor(np.asarray(ObjectWrapper(tensor_acc_list[0].numpy())))
