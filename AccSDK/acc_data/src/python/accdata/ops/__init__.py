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

import accdata.backend as _b
from accdata.data_node import DataNode as _DataNode
from accdata.pipeline import Pipeline as _Pipeline
from accdata.types import TensorLayout


class Operator:
    def __init__(self, name):
        self.spec = _b.new_op(name)
        self.output = None

    def add_input(self, input_arg):
        if not isinstance(input_arg, _DataNode):
            raise TypeError(f"Expected inputs of type 'DataNode'. Received input of type '{type(input_arg)}'.")
        self.spec.AddInput(input_arg.name, input_arg.device)
        return self

    def add_output(self, name, device="cpu"):
        self.output = _Pipeline.get_current().create_datanode(name, device)
        self.spec.AddOutput(self.output.name, self.output.device)
        return self

    def add_arg(self, key, value, types, cast=None, count=None, choice=None):
        if value not in (0, False) and not value:
            return self

        if count:
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Expected '{key}' argument of list of type '{types}', received {type(value)}")

        if isinstance(value, (list, tuple)):
            if not all(isinstance(v, type(value[0])) for v in value):
                raise TypeError(f"Type of elements should be the same.")

            if not isinstance(value[0], types):
                raise TypeError(f"Expected '{key}' argument of type '{types}', received {type(value[0])}")

            if choice and not all(v in choice for v in value):
                raise ValueError(f"Expected '{choice}', received '{value}'")

            if count and len(value) != count:
                raise ValueError(f"Expect '{count}' elements, received '{len(value)}'")

            self.spec.AddArg(key, value if not cast else [cast(v) for v in value])
            return self

        if not isinstance(value, types):
            raise TypeError(f"Expected '{key}' argument of type '{types}', received {type(value)}")

        if choice and value not in choice:
            raise ValueError(f"Expected '{choice}', received '{value}'")

        if count:
            ext_value = []
            for _ in range(count):
                ext_value.append(value if not cast else cast(value))
            self.spec.AddArg(key, ext_value)
            return self

        self.spec.AddArg(key, value if not cast else cast(value))
        return self


ARG_RESIZE = "resize"
ARG_INTERPOLATION_MODE = "interpolation_mode"
INTERPOLATION_MODE_BILINEAR = "bilinear"
INTERPOLATION_MODE_BICUBIC = "bicubic"
ARG_CROP_POS_X = "crop_pos_x"
ARG_CROP_POS_Y = "crop_pos_y"
ARG_CROP = "crop"
ARG_ROUND_MODE = "round_mode"
ROUND_MODE_ROUND = "round"
ROUND_MODE_TRUNCATE = "truncate"
ARG_MEAN = "mean"
ARG_STDDEV = "stddev"
ARG_SCALE = "scale"
ARG_LAYOUT = "layout"
NODE_EXTERNAL_SOURCE = "ExternalSource"
NODE_TO_TENSOR = "ToTensor"
NODE_NORMALIZE = "Normalize"
NODE_RESIZE_CROP = "ResizeCrop"
NODE_TO_TENSOR_RESIZE_CROP_NORMALIZE = "ToTensorResizeCropNormalize"
RGB_CHANNELS = 3


def external_source(name: str):
    if not isinstance(name, str):
        raise TypeError(f"Expected inputs of type 'str'. Received input of type '{type(name)}'.")
    return Operator(NODE_EXTERNAL_SOURCE)\
                    .add_output(name)


def to_tensor(input_tensor, layout=TensorLayout.NCHW):
    return Operator(NODE_TO_TENSOR)\
                    .add_input(input_tensor)\
                    .add_arg(ARG_LAYOUT, layout, (TensorLayout), cast=int)\
                    .add_output(NODE_TO_TENSOR)


def resize_crop(
        input_tensor,
        resize,
        crop=None,
        interpolation_mode="bilinear",
        round_mode="truncate"):
    """
    resize crop operation
    :param input_tensor:  input data,layout should be nchw while dtype is fp32,nhwc is unsupported for fusion operator
    :param resize: (height, width) tuple for resize target size, if crop is None ,this is the final image size
    :param crop:   (height, width) tuple for final crop size
    :param resize_w: The width of the resized image.
    :param resize_h: The height of the resized image.
    :param crop_w: Cropping the window width(in pixels).
    :param crop_h: Cropping the window height(in pixels).
    :param interpolation_mode: interpolation mode
    :return:
    """
    if resize is None or not isinstance(resize, (list, tuple)) or len(resize) != 2 or \
        not all(isinstance(x, int) for x in resize):
        raise ValueError("Invalid resize input!")
    if crop:
        # crop is supposed to be list or tuple with all int elements and length 2
        if not isinstance(crop, (list, tuple)) or len(crop) != 2 or not all(isinstance(x, int) for x in crop):
            raise ValueError("Invalid crop input!")
        # crop size should not larger than resize size
        if crop[0] > resize[0]:
            raise ValueError("crop height size cannot greater than resize height size!")
        if crop[1] > resize[1]:
            raise ValueError("crop width size cannot greater than resize width size!")

    return Operator(NODE_RESIZE_CROP)\
                    .add_input(input_tensor)\
                    .add_arg(ARG_RESIZE, resize, (int), cast=int, count=2)\
                    .add_arg(ARG_CROP, crop if crop else resize, (int),
                             cast=int, count=2)\
                    .add_arg(ARG_INTERPOLATION_MODE, interpolation_mode, (str),
                             choice=(INTERPOLATION_MODE_BILINEAR, INTERPOLATION_MODE_BICUBIC))\
                    .add_arg(ARG_CROP_POS_X, 0.5, float)\
                    .add_arg(ARG_CROP_POS_Y, 0.5, float)\
                    .add_arg(ARG_ROUND_MODE, round_mode, (str), choice=(ROUND_MODE_ROUND, ROUND_MODE_TRUNCATE))\
                    .add_output(NODE_RESIZE_CROP)


def normalize(input_tensor, mean, std, scale=None):
    """
    Normalizes the input by removing the mean and dividing by the standard deviation.
        Formula: out = (in - mean) / stddev * scale
    :param mean: Mean value to be subtracted from the data.
    :param stddev: Standard deviation value to scale the data.
    :param scale: The scaling factor applied to the output. Default is 1.0.
    """
    return Operator(NODE_NORMALIZE)\
                    .add_input(input_tensor)\
                    .add_arg(ARG_MEAN, mean, (float), count=RGB_CHANNELS)\
                    .add_arg(ARG_STDDEV, std, (float), count=RGB_CHANNELS)\
                    .add_arg(ARG_SCALE, scale, (float))\
                    .add_output(NODE_NORMALIZE)


def to_tensor_resize_crop_norm(
        input_tensor,
        resize=None, interpolation_mode=INTERPOLATION_MODE_BILINEAR,  # resize params
        crop_pos_w=0.5, crop_pos_h=0.5, crop=None, round_mode=None,  # crop params
        mean=None, std=None, scale=None,  # norm params
        layout=TensorLayout.NCHW  # to_tensor params
):
    if resize is None or not isinstance(resize, (list, tuple)) or len(resize) != 2 or \
        not all(isinstance(x, int) for x in resize):
        raise ValueError("Invalid resize input!")
    if crop:
        # crop is supposed to be list or tuple with all int elements and length 2
        if not isinstance(crop, (list, tuple)) or len(crop) != 2 or not all(isinstance(x, int) for x in crop):
            raise ValueError("Invalid crop input!")
        # crop size should not larger than resize size
        if crop[0] > resize[0]:
            raise ValueError("crop height size cannot greater than resize height size!")
        if crop[1] > resize[1]:
            raise ValueError("crop width size cannot greater than resize width size!")

    return Operator(NODE_TO_TENSOR_RESIZE_CROP_NORMALIZE)\
                    .add_input(input_tensor)\
                    .add_arg(ARG_RESIZE, resize, (int), cast=int, count=2)\
                    .add_arg(ARG_INTERPOLATION_MODE, interpolation_mode, (str), choice=(INTERPOLATION_MODE_BILINEAR))\
                    .add_arg(ARG_CROP_POS_X, crop_pos_w, (float))\
                    .add_arg(ARG_CROP_POS_Y, crop_pos_h, (float))\
                    .add_arg(ARG_CROP, crop if crop else resize, (int), cast=int, count=2)\
                    .add_arg(ARG_ROUND_MODE, round_mode, (str), choice=(ROUND_MODE_ROUND, ROUND_MODE_TRUNCATE))\
                    .add_arg(ARG_MEAN, mean, (float))\
                    .add_arg(ARG_STDDEV, std, (float))\
                    .add_arg(ARG_SCALE, scale, (float))\
                    .add_arg(ARG_LAYOUT, layout, (TensorLayout), cast=int)\
                    .add_output(NODE_NORMALIZE)


def qwen_fusion_op(input_tensor, mean, std, min_pixels=56*56, max_pixels=28*28*1280, patch_size=14,
        temporal_patch_size=2, merge_size=2):
    return Operator("QwenFusionOp")\
                    .add_input(input_tensor)\
                    .add_arg(ARG_MEAN, mean, (float))\
                    .add_arg(ARG_STDDEV, std, (float))\
                    .add_arg("min_pixels", min_pixels, (int))\
                    .add_arg("max_pixels", max_pixels, (int))\
                    .add_arg("patch_size", patch_size, (int))\
                    .add_arg("temporal_patch_size", temporal_patch_size, (int))\
                    .add_arg("merge_size", merge_size, (int))\
                    .add_output("qwen_output")
