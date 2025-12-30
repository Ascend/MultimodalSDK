#!/usr/bin/python3
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
import random
import logging

import torch
import pytest
from torchvision.transforms import functional as F

import accdata.types as _t

MIN_HW = 10
MAX_HW = 8192


class HandleSource:
    def __init__(self, source_shape, target_size, data_type, layout, name):
        if data_type == torch.float:
            self.tensor = torch.rand(source_shape, dtype=torch.float)
        else:
            self.tensor = torch.randint(0, 256, source_shape, dtype=torch.uint8)
        self.target_size = target_size
        self.name = name
        self.source_shape = source_shape
        if (layout == _t.TensorLayout.NHWC):
            self.origin_size = (source_shape[1], source_shape[2])
        else:
            self.origin_size = (source_shape[2], source_shape[3])
        self.layout = layout

    def get_resize_pos(self):
        rh, rw = self.target_size[0] / \
                 self.origin_size[0], self.target_size[1] / self.origin_size[1]
        if rh > rw:
            resize_pos = self.target_size[0], round(
                self.origin_size[1] * rh)
        else:
            resize_pos = round(
                self.origin_size[0] * rw), self.target_size[1]
            
        return resize_pos

RANDOM_SHAPE_COUNT = 1000


def get_random_shape():
    return [random.randint(MIN_HW, MAX_HW), random.randint(MIN_HW, MAX_HW)]


random_input_shape = [get_random_shape() for _ in range(RANDOM_SHAPE_COUNT)]
random_output_shape = [get_random_shape() for _ in range(RANDOM_SHAPE_COUNT)]


class RandomDataSource():
    data_uint8_nhwc = [
        HandleSource((1, 1080, 1920, 3), (1024, 1024), torch.uint8, _t.TensorLayout.NHWC, "1080p"),
        HandleSource((2, 1080, 1920, 3), (1024, 1024), torch.uint8, _t.TensorLayout.NHWC, "1080p_bs2"),
        HandleSource((1, 720, 1280, 3), (512, 512), torch.uint8, _t.TensorLayout.NHWC, "720p"),
    ]
    data_uint8_nchw = [
        HandleSource((1, 3, 1920, 1080), (1024, 1024), torch.uint8, _t.TensorLayout.NCHW, "1080p"),
        HandleSource((2, 3, 1920, 1080), (1024, 1024), torch.uint8, _t.TensorLayout.NCHW, "1080p_bs2"),
        HandleSource((1, 3, 1280, 720), (512, 512), torch.uint8, _t.TensorLayout.NCHW, "720p"),
    ]
    data_float_nhwc = [
        HandleSource((1, 1080, 1920, 3), (1024, 1024), torch.float, _t.TensorLayout.NHWC, "1080p"),
        HandleSource((2, 1080, 1920, 3), (1024, 1024), torch.float, _t.TensorLayout.NHWC, "1080p_bs2"),
        HandleSource((1, 720, 1280, 3), (512, 512), torch.float, _t.TensorLayout.NHWC, "720p"),
    ]
    data_float_nchw = [
        HandleSource((1, 3, 1920, 1080), (1024, 1024), torch.float, _t.TensorLayout.NCHW, "1080p"),
        HandleSource((2, 3, 1920, 1080), (1024, 1024), torch.float, _t.TensorLayout.NCHW, "1080p_bs2"),
        HandleSource((1, 3, 1280, 720), (512, 512), torch.float, _t.TensorLayout.NCHW, "720p"),
    ]

    data_uint8_nhwc_random_shape = data_uint8_nhwc.copy()
    data_uint8_nchw_random_shape = data_uint8_nchw.copy()
    data_float_nhwc_random_shape = data_float_nhwc.copy()
    data_float_nchw_random_shape = data_float_nchw.copy()

    @staticmethod
    def create_random_data_source_slow():
        if len(RandomDataSource.data_uint8_nhwc_random_shape) == RANDOM_SHAPE_COUNT:
            return
        RandomDataSource.data_uint8_nhwc_random_shape = [
            HandleSource([1] + random_input_shape[i] + [3],
                        random_output_shape[i], torch.uint8, _t.TensorLayout.NHWC, "random_" + str(i))
            for i in range(RANDOM_SHAPE_COUNT)
        ]
        RandomDataSource.data_uint8_nchw_random_shape = [
            HandleSource([1, 3] + random_input_shape[i],
                        random_output_shape[i], torch.uint8, _t.TensorLayout.NCHW, "random_" + str(i))
            for i in range(RANDOM_SHAPE_COUNT)
        ]
        RandomDataSource.data_float_nhwc_random_shape = [
            HandleSource([1] + random_input_shape[i] + [3],
                        random_output_shape[i], torch.float, _t.TensorLayout.NHWC, "random_" + str(i))
            for i in range(RANDOM_SHAPE_COUNT)
        ]
        RandomDataSource.data_float_nchw_random_shape = [
            HandleSource([1, 3] + random_input_shape[i],
                        random_output_shape[i], torch.float, _t.TensorLayout.NCHW, "random_" + str(i))
            for i in range(RANDOM_SHAPE_COUNT)
        ]


class TorchOpTransforms():

    def __init__(self):
        pass

    @staticmethod
    def resize(clip, target_size, interpolation_mode):
        if len(target_size) != 2:
            raise ValueError(
                f"target size should be tuple (height, width), instead got {target_size}"
            )
        return torch.nn.functional.interpolate(clip,
                                               size=target_size,
                                               mode=interpolation_mode,
                                               align_corners=False)

    @staticmethod
    def crop(clip, i, j, h, w):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        """
        if len(clip.size()) != 4:
            raise ValueError("clip should be a 4D tensor")
        return clip[..., i:i + h, j:j + w]

    @staticmethod
    def resize_crop(clip, args, resize_pos=None):
        h, w = clip.size(-2), clip.size(-1)
        th, tw = args.target_size[0], args.target_size[1]
        rh, rw = th / h, tw / w
        if rh > rw:
            if resize_pos:
                sh, sw = resize_pos[0], resize_pos[1]
            else:
                sh, sw = th, round(w * rh)
            clip = TorchOpTransforms.resize(clip, (sh, sw), args.interpolation_mode)
            i = int(round(sh - th) / 2.0)
            j = int(round(sw - tw) / 2.0)
        else:
            if resize_pos:
                sh, sw = resize_pos[0], resize_pos[1]
            else:
                sh, sw = round(h * rw), tw
            clip = TorchOpTransforms.resize(clip, (sh, sw), args.interpolation_mode)
            i = int(round(sh - th) / 2.0)
            j = int(round(sw - tw) / 2.0)
        result = TorchOpTransforms.crop(clip, i, j, th, tw)

        return result

    @staticmethod
    def normalize(clip, args):
        clip = F.normalize(clip, mean=args.mean, std=args.std)
        if hasattr(args, "scale"):
            clip = args.scale * clip
        return clip


    @staticmethod
    def to_tensor_with_layout(clip, src_layout, target_layout):
        if not clip.dtype == torch.uint8:
            raise TypeError("clip tensor should have data type uint8. Got %s" %
                            str(clip.dtype))
        if src_layout == target_layout:
            clip = clip.float() / 255.0
            return clip

        if target_layout == _t.TensorLayout.NCHW:
            clip = clip.float() / 255.0
            clip = clip.permute(0, 3, 1, 2)
        if target_layout == _t.TensorLayout.NHWC:
            clip = clip.float() / 255.0
            clip = clip.permute(0, 2, 3, 1)

        return clip


    @staticmethod
    def to_tensor(clip):
        return TorchOpTransforms.to_tensor_with_layout(clip, _t.TensorLayout.NHWC, _t.TensorLayout.NCHW)

    @staticmethod
    def to_tensor_resize_crop(clip, args, resize_pos=None):
        to_tesnor = TorchOpTransforms.to_tensor(clip)
        resize_crop = TorchOpTransforms.resize_crop(to_tesnor, args, resize_pos=resize_pos)
        return resize_crop

    @staticmethod
    def to_tensor_resize_crop_normalize(clip, args, resize_pos=None):
        to_tesnor = TorchOpTransforms.to_tensor(clip)
        resize_crop = TorchOpTransforms.resize_crop(to_tesnor, args, resize_pos=resize_pos)
        normalize = TorchOpTransforms.normalize(resize_crop, args)
        return normalize

    @staticmethod
    def check_image_size(image_size):
        for image_length in image_size:
            if image_length < MIN_HW or image_length > MAX_HW:
                return False
        return True

    @staticmethod
    def compare_tensors(tensor1, tensor2, relative_tol=1e-4, absolute_tol=1e-5, error_tol=1e-4):
        if torch.equal(tensor1, tensor2):
            return True
        compare_tensor = torch.isclose(tensor1, tensor2, atol=absolute_tol, rtol=relative_tol)
        if torch.all(compare_tensor):
            return True

        correct_count = float(compare_tensor.sum())
        ele_count = compare_tensor.numel()
        err_ratio = (ele_count - correct_count) / ele_count
        if err_ratio >= error_tol:
            tensor1 = tensor1.clone().reshape(-1)
            tensor2 = tensor2.clone().reshape(-1)
            compare_tensor = torch.isclose(tensor1, tensor2, atol=absolute_tol, rtol=relative_tol)
            different_element_indexes = torch.where(compare_tensor == False)[0]
            for count, real_index in enumerate(different_element_indexes):
                tensor1_data = tensor1[real_index]
                tensor2_data = tensor2[real_index]
                logging.error(
                    "data index: %06d, expected: %-.9f, actual: %-.9f, adiff: %-.6f" %
                    (real_index, tensor1_data, tensor2_data,
                    abs(tensor1_data - tensor2_data)))
                if count == 100:
                    break
            logging.error(
                    "expected error ratio: %-.9f, actual: %-.9f" %
                    (error_tol, err_ratio))

        return err_ratio < error_tol
