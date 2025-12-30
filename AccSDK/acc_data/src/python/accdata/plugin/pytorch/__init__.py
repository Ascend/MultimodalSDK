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

from typing import List, Union

import torch
import accdata.backend as _b
import accdata.types as _types

to_torch_type = {
    _types.TensorDataType.FP32: torch.float,
    _types.TensorDataType.UINT8: torch.uint8,
}


def to_torch_tensor(t: _b.Tensor) -> torch.tensor:
    """
    transform AccData Tensor to torch tensor using share buffer
    """
    if not isinstance(t, _b.Tensor):
        raise TypeError("Only Accdata Tensor supported.")

    view = memoryview(t)
    if view.format not in ["=f", "=b"]:
        raise TypeError(
            "Only torch.float32 or torch.uint8 can transform to Accdata Tensor.")

    return torch.as_strided(
        input=torch.frombuffer(
            t, dtype=torch.float if view.format == "=f" else torch.uint8),
        size=view.shape,
        stride=[s // view.strides[-1] for s in view.strides]
    )


def to_torch_tensorlist(tl: _b.TensorList) -> List[torch.tensor]:
    """
    transform AccData TensorLists to a list of torch tensor using share buffer
    """
    if not isinstance(tl, _b.TensorList):
        raise TypeError("Only list of torch tensors supported.")

    output_list = []
    for i in range(len(tl)):
        elm = tl[i]
        output_list.append(to_torch_tensor(elm))
    return output_list


def to_accdata_tensorlist(tl: List[torch.tensor], layout=None) -> _b.TensorList:
    """
    transform a list of torch tensors to AccData TensorList using share buffer

    @:param tl      the list of tensors need to convert to Accdata TensorList
    @:param layout  specify the origin layout of the tensor, i.e. without permute operation.
                    Generally, it's hard for developer to track the origin layout. Thus, in the commonly used RGB
                    image processing scene, we try to set the layout for user:
                       - if the second dimension is channel(=3 in commonly used RGB scene), set the layout as 'NCHW'
                       - if the last dimension is channel, the layout is set as 'NHWC'.
    """
    if not isinstance(tl, list):
        raise TypeError("Only list of torch tensors supported.")
    for tensor in tl:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Only torch tensor supported.")
    result = _b.new_tensorlist(len(tl))
    for idx, torch_tensor in enumerate(tl):
        # sort according to strides info to get origin data shape
        sorted_tuples = sorted(
            zip(torch_tensor.stride(), torch_tensor.shape), reverse=True)
        _, shape = zip(*sorted_tuples)
        # set layout for 4-D Tensor in the RGB image processing scene
        if layout is None and len(shape) == 4:
            if shape[1] == 3:
                layout = _types.TensorLayout.NCHW
            elif shape[-1] == 3:
                layout = _types.TensorLayout.NHWC
            else:
                raise TypeError("Only NCHW and NHWC supported.")
        dtype = _types.TensorDataType.UINT8 if torch_tensor.dtype == torch.uint8 else _types.TensorDataType.FP32
        result[idx].ShareTorchData(torch_tensor.numpy(), shape, layout, dtype)

    return result