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
import torch
from typing import Union, List, Dict, Tuple, Optional
import PIL.Image as PILImage

from ..acc.wrapper import Image, DeviceMode, Interpolation, TensorFormat

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_IMAGE_SIZE = 8192
MIN_IMAGE_SIZE = 10


def _get_target_ratios(
        min_num: int,
        max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = []
    for i in range(1, max_num + 1):
        for j in range(1, max_num + 1):
            product = i * j
            if min_num <= product <= max_num:
                target_ratios.append((i, j))
    target_ratios = sorted(set(target_ratios), key=lambda x: x[0] * x[1])
    return target_ratios


def _find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        *,
        width: int,
        height: int,
        image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _calculate_target_dimensions(
        *,
        orig_width: int,
        orig_height: int,
        target_ratios: list[tuple[int, int]],
        image_size: int,
        use_thumbnail: bool,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height
    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks, target_width, target_height


class InternVL2PreProcessor:

    @staticmethod
    def preprocess_image(
            image: Union[PILImage.Image, Image],
            input_size: int,
            min_num: int,
            max_num: int,
            use_thumbnail: bool,
    ) -> torch.Tensor:
        """
        InternVL2 preprocess function

        Args:
            image Union[PILImage.Image, Image]: given image, either a Pillow Image or the internal Image type.
            input_size int: Target size for each patch. The image will be resized and split
            min_num int: Minimum number of patches used to compute target ratios.
            max_num int: Maximum number of patches used to compute target ratios.
            use_thumbnail bool: Whether to generate an additional thumbnail patch of size

        Returns:
            Tensor: after preprocess
        """
        if not isinstance(image, (PILImage.Image, Image)):
            raise TypeError(f"image must be PIL.Image.Image or multi modal Image.")

        if min_num <= 0 or min_num > 4 or min_num > max_num:
            raise ValueError("The input param 'min_num' must be in range [1, 4] and less than max_num, please check.")

        if max_num < min_num or max_num > 32:
            raise ValueError("The input param 'max_num' must be greater than min_num and less than 32, please check.")

        if isinstance(image, PILImage.Image):
            image = Image.from_pillow(image)

        orig_width = image.width
        orig_height = image.height
        if (orig_width > MAX_IMAGE_SIZE or orig_height > MAX_IMAGE_SIZE or
                orig_width < MIN_IMAGE_SIZE or orig_height < MIN_IMAGE_SIZE):
            raise ValueError("The input image size must in range [10, 8192].")

        if input_size < 10 or input_size > 8192:
            raise ValueError("The input param 'input_size' must be in range [10, 8192], please check.")


        target_ratios = _get_target_ratios(min_num, max_num)

        blocks, target_width, target_height = _calculate_target_dimensions(
            orig_width=orig_width,
            orig_height=orig_height,
            target_ratios=target_ratios,
            image_size=input_size,
            use_thumbnail=False,
        )

        resized_img = image.resize(
            (target_width, target_height), Interpolation.BICUBIC, DeviceMode.CPU
        )
        processed_tensors = []
        for i in range(blocks):
            left = (i % (target_width // input_size)) * input_size
            top = (i // (target_width // input_size)) * input_size
            right = ((i % (target_width // input_size)) + 1) * input_size
            bottom = ((i // (target_width // input_size)) + 1) * input_size
            width = right - left
            height = bottom - top
            # split the image
            split_img = resized_img.crop(top, left, height, width, DeviceMode.CPU)
            tensor = split_img.to_tensor(TensorFormat.NCHW)
            tensor = tensor.normalize(IMAGENET_MEAN, IMAGENET_STD)
            processed_tensors.append(tensor)

        if use_thumbnail and len(processed_tensors) != 1:
            thumbnail_img = image.resize(
                (input_size, input_size), Interpolation.BICUBIC, DeviceMode.CPU
            )
            thumbnail_tensor = thumbnail_img.to_tensor(TensorFormat.NCHW)
            thumbnail_tensor = thumbnail_tensor.normalize(IMAGENET_MEAN, IMAGENET_STD)
            processed_tensors.append(thumbnail_tensor)

        pixel_values = torch.stack([tensor.torch().squeeze(0) for tensor in processed_tensors])
        return pixel_values
