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
import os
import sys
from PIL import Image as PImage
from torchvision import transforms
import numpy as np
import unittest
import torch

import mm
from mm import Image, DataType, ImageFormat, DeviceMode, TensorFormat
from test_tensor import compare_float_torch_tensors

IM_WIDTH = 10
IM_HEIGHT = 10
INVALID_IM_MIN_WIDTH = 9
INVALID_IM_MAX_WIDTH = 8193
INVALID_IM_MIN_HEIGHT = 9
INVALID_IM_MAX_HEIGHT = 8193
THREE_CHANNEL = 3
FOUR_CHANNEL = 4
SUPPORT_MODE = "RGB"
UNSUPPORT_MODE = "RGBA"
DEVICE_CPU = "cpu"
WIDTH_960 = 960
HEIGHT_840 = 840
CROP_WIDTH = 20
CROP_HEIGHT = 30
RESIZE_HEIGHT = 40
RESIZE_WIDTH = 50
INVALID_HEIGHT = 8200
INVALID_WIDTH = 8200


def random_hw():
    H = np.random.randint(10, 8193)
    W = np.random.randint(10, 8193)
    return H, W


H_packed, W_packed = random_hw()
PACKED_ARRAY = np.random.randint(0, 256, size=(H_packed, W_packed, 3), dtype=np.uint8)
H_planar, W_planar = random_hw()
PLANAR_ARRAY = np.random.randint(0, 256, size=(3, H_planar, W_planar), dtype=np.uint8)
TEST_ARRAY_INT8 = PACKED_ARRAY.astype(np.int8)
TEST_ARRAY_FLOAT32 = PACKED_ARRAY.astype(np.float32)
TEST_ARRAY_FLOAT16 = PACKED_ARRAY.astype(np.float16)
TEST_TORCH_IMAGE_UINT8 = torch.tensor(PACKED_ARRAY, dtype=torch.uint8)
PACKED_TORCH = torch.randint(0, 256, (H_packed, W_packed, 3), dtype=torch.uint8)
PLANAR_TORCH = torch.randint(0, 256, (3, H_planar, W_planar), dtype=torch.uint8)
TEST_TORCH_INT8 = PACKED_TORCH.to(torch.int8)
TEST_TORCH_FLOAT32 = PACKED_TORCH.to(torch.float32)
TEST_TORCH_FLOAT16 = PACKED_TORCH.to(torch.float16)


def array_full_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return (
            a.shape == b.shape and
            a.dtype == b.dtype and
            a.flags['C_CONTIGUOUS'] == b.flags['C_CONTIGUOUS'] and
            a.flags['F_CONTIGUOUS'] == b.flags['F_CONTIGUOUS'] and
            a.strides == b.strides and
            np.array_equal(a, b)
    )


def tensor_full_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
            a.shape == b.shape and
            a.dtype == b.dtype and
            a.device == b.device and
            a.stride() == b.stride() and
            a.is_contiguous() == b.is_contiguous() and
            torch.equal(a, b)
    )


class TestImage(unittest.TestCase):
    valid_path = None
    invalid_path = None

    @classmethod
    def setUpClass(cls):
        cls.valid_path = "./test/assets/dog_1920_1080.jpg"
        cls.invalid_path = "./test/assets/dog_1920_1080.png"

    def setUp(self):
        self.arr = np.random.randint(0, 255, (IM_HEIGHT, IM_WIDTH, THREE_CHANNEL), dtype=np.uint8)
        self.pillow_image = PImage.fromarray(self.arr, mode=SUPPORT_MODE)

        self.arr_rgba = np.random.randint(0, 255, (IM_HEIGHT, IM_WIDTH, FOUR_CHANNEL), dtype=np.uint8)
        self.pillow_image_rgba = PImage.fromarray(self.arr_rgba, mode=UNSUPPORT_MODE)

        self.arr_invalid_min_height = np.random.randint(0, 255, (INVALID_IM_MIN_HEIGHT, IM_WIDTH, THREE_CHANNEL),
                                                        dtype=np.uint8)
        self.pillow_image_invalid_min_height = PImage.fromarray(self.arr_invalid_min_height, mode=SUPPORT_MODE)

        self.arr_invalid_max_height = np.random.randint(0, 255, (INVALID_IM_MAX_HEIGHT, IM_WIDTH, THREE_CHANNEL),
                                                        dtype=np.uint8)
        self.pillow_image_invalid_max_height = PImage.fromarray(self.arr_invalid_max_height, mode=SUPPORT_MODE)

        self.arr_invalid_min_width = np.random.randint(0, 255, (IM_HEIGHT, INVALID_IM_MIN_WIDTH, THREE_CHANNEL),
                                                       dtype=np.uint8)
        self.pillow_image_invalid_min_width = PImage.fromarray(self.arr_invalid_min_width, mode=SUPPORT_MODE)

        self.arr_invalid_max_width = np.random.randint(0, 255, (IM_HEIGHT, INVALID_IM_MAX_WIDTH, THREE_CHANNEL),
                                                       dtype=np.uint8)
        self.pillow_image_invalid_max_width = PImage.fromarray(self.arr_invalid_max_width, mode=SUPPORT_MODE)

    def test_create_empty_image_should_fail(self):
        is_failed = False
        try:
            image = mm.Image()
        except Exception as e:
            is_failed = True
        self.assertTrue(is_failed)

    def test_create_image_from_invalid_path_should_fail(self):
        is_failed = False
        try:
            os.chmod(self.invalid_path, 0o640)
            image = mm.Image.open(self.invalid_path, DEVICE_CPU)
        except Exception as e:
            is_failed = True
        self.assertTrue(is_failed)

    def test_create_image_from_valid_path_with_wrong_permission_should_fail(self):
        is_failed = False
        try:
            os.chmod(self.valid_path, 0o644)
            image = mm.Image.open(self.valid_path, DEVICE_CPU)
        except Exception as e:
            is_failed = True
        self.assertTrue(is_failed)

    def test_create_image_from_valid_path_with_should_success(self):
        is_failed = False
        try:
            os.chmod(self.valid_path, 0o640)
            image = mm.Image.open(self.valid_path, DEVICE_CPU)
            self.assertEqual(list(image.size), [1920, 1080])
            self.assertEqual(image.height, 1080)
            self.assertEqual(image.width, 1920)
            self.assertEqual(image.format.value, 12)
            self.assertEqual(image.dtype.value, 4)
            self.assertEqual(image.device, DEVICE_CPU)
        except Exception as e:
            self.fail(f"Expected no exception, but got {e}")

    def test_create_image_from_clone_should_success(self):
        is_failed = False
        try:
            os.chmod(self.valid_path, 0o640)
            image = mm.Image.open(self.valid_path, DEVICE_CPU)
            self.assertEqual(list(image.size), [1920, 1080])
            self.assertEqual(image.height, 1080)
            self.assertEqual(image.width, 1920)
            self.assertEqual(image.format.value, 12)
            self.assertEqual(image.dtype.value, 4)
            self.assertEqual(image.device, DEVICE_CPU)
            image_clone = image.clone()
            self.assertEqual(list(image_clone.size), [1920, 1080])
            self.assertEqual(image_clone.height, 1080)
            self.assertEqual(image_clone.width, 1920)
            self.assertEqual(image_clone.format.value, 12)
            self.assertEqual(image_clone.dtype.value, 4)
            self.assertEqual(image_clone.device, DEVICE_CPU)
        except Exception as e:
            self.fail(f"Expected no exception, but got {e}")

    def test_numpy_to_image_success_uint8_packed(self):
        image = Image.from_numpy(PACKED_ARRAY, ImageFormat.RGB, DEVICE_CPU)
        self.assertEqual(image.dtype, DataType.UINT8)
        self.assertEqual(image.device, DEVICE_CPU)
        self.assertEqual(image.size, [PACKED_ARRAY.shape[1], PACKED_ARRAY.shape[0]])  # W,H
        self.assertEqual(image.format, ImageFormat.RGB)
        self.assertEqual(image.nbytes, PACKED_ARRAY.nbytes)
        arr = image.numpy()
        self.assertTrue(array_full_equal(arr, PACKED_ARRAY))

    def test_numpy_to_image_success_uint8_planar(self):
        image = Image.from_numpy(PLANAR_ARRAY, ImageFormat.RGB_PLANAR, DEVICE_CPU)
        self.assertEqual(image.size, [PLANAR_ARRAY.shape[2], PLANAR_ARRAY.shape[1]])  # W,H
        self.assertEqual(image.format, ImageFormat.RGB_PLANAR)
        arr = image.numpy()
        self.assertTrue(array_full_equal(arr, PLANAR_ARRAY))

    def test_numpy_to_image_fail_with_wrong_input_type(self):
        with self.assertRaises(TypeError) as context:
            Image.from_numpy(list(PLANAR_ARRAY), ImageFormat.RGB)
        self.assertEqual(
            str(context.exception),
            "The input param 'nd_array' must be of numpy's ndarray type."
        )

    def test_numpy_to_image_fail_with_no_contiguous(self):
        with self.assertRaises(ValueError) as context:
            Image.from_numpy(PACKED_ARRAY[:, ::-1, :], ImageFormat.RGB)
        self.assertIn("must be c_contiguous", str(context.exception))

    def test_numpy_to_image_fail_with_wrong_dtype(self):
        wrong_dtypes = [
            (TEST_ARRAY_INT8, np.int8),
            (TEST_ARRAY_FLOAT32, np.float32),
            (TEST_ARRAY_FLOAT16, np.float16)
        ]
        for arr, dtype in wrong_dtypes:
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError) as context:
                    Image.from_numpy(arr, ImageFormat.RGB)
                self.assertEqual(
                    str(context.exception),
                    "The input numpy's ndarray data type must be np.uint8"
                )

    def test_torch_to_image_success_uint8_packed(self):
        image = Image.from_torch(PACKED_TORCH, ImageFormat.RGB, DEVICE_CPU)
        self.assertEqual(image.dtype, DataType.UINT8)
        self.assertEqual(image.device, DEVICE_CPU)
        self.assertEqual(image.size, [PACKED_TORCH.shape[1], PACKED_TORCH.shape[0]])
        self.assertEqual(image.format, ImageFormat.RGB)
        tensor_back = image.torch()
        self.assertTrue(tensor_full_equal(tensor_back, PACKED_TORCH))

    def test_torch_to_image_success_uint8_planar(self):
        image = Image.from_torch(PLANAR_TORCH, ImageFormat.RGB_PLANAR, DEVICE_CPU)
        self.assertEqual(image.size, [PLANAR_TORCH.shape[2], PLANAR_TORCH.shape[1]])
        self.assertEqual(image.format, ImageFormat.RGB_PLANAR)

        tensor_back = image.torch()
        self.assertTrue(tensor_full_equal(tensor_back, PLANAR_TORCH))

    def test_torch_to_image_fail_with_no_contiguous(self):
        with self.assertRaises(ValueError) as context:
            Image.from_torch(PACKED_TORCH.T, ImageFormat.RGB)
        self.assertIn("must be c_contiguous", str(context.exception))

    def test_torch_to_image_fail_with_wrong_dtype(self):
        wrong_dtypes = [
            (TEST_TORCH_INT8, torch.int8),
            (TEST_TORCH_FLOAT32, torch.float32),
            (TEST_TORCH_FLOAT16, torch.float16)
        ]
        for t, dtype in wrong_dtypes:
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError) as context:
                    Image.from_torch(t, ImageFormat.RGB)
                self.assertEqual(
                    str(context.exception),
                    "The input torch tensor data type must be torch.uint8"
                )

    def test_torch_to_image_fail_with_wrong_input_type(self):
        with self.assertRaises(TypeError) as context:
            Image.from_torch(np.zeros((10, 10, 3), dtype=np.uint8), ImageFormat.RGB)
        self.assertEqual(
            str(context.exception),
            "The parameter 'torch_tensor' must be of torch.Tensor type."
        )

    def test_pillow_to_image_shoule_success(self):
        image = mm.Image.from_pillow(self.pillow_image)
        self.assertEqual(image.size, [IM_WIDTH, IM_HEIGHT])
        self.assertEqual(image.width, IM_WIDTH)
        self.assertEqual(image.height, IM_HEIGHT)
        self.assertEqual(image.nbytes, self.arr.size * self.arr.itemsize)
        self.assertEqual(image.format, mm.ImageFormat.RGB)
        self.assertEqual(image.dtype, mm.DataType.UINT8)
        self.assertEqual(image.device, DEVICE_CPU)

    def test_pillow_to_image_should_fail_with_invalid_pillow_mode(self):
        with self.assertRaises(ValueError) as context:
            mm.Image.from_pillow(self.pillow_image_rgba)
        expected_message = "The input pillow's Image mode must be in ['RGB']"
        self.assertEqual(str(context.exception), expected_message)

    def test_pillow_to_image_should_fail_with_unexpected_input(self):
        with self.assertRaises(TypeError) as context:
            mm.Image.from_pillow(self.arr)
        expected_message = "The parameter 'pillow_image' must be of PIL.Image.Image type."
        self.assertEqual(str(context.exception), expected_message)

    def test_pillow_to_image_should_fail_with_unexpected_size(self):
        with self.assertRaises(RuntimeError) as context:
            mm.Image.from_pillow(self.pillow_image_invalid_min_width)
        self.assertIn("image size or format is invalid", str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            mm.Image.from_pillow(self.pillow_image_invalid_max_width)
        self.assertIn("image size or format is invalid", str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            mm.Image.from_pillow(self.pillow_image_invalid_max_height)
        self.assertIn("image size or format is invalid", str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            mm.Image.from_pillow(self.pillow_image_invalid_min_height)
        self.assertIn("image size or format is invalid", str(context.exception))

    def test_image_to_pillow_should_success(self):
        image = mm.Image.from_pillow(self.pillow_image)
        pillow_image = image.pillow()
        self.assertEqual(pillow_image.size, (IM_WIDTH, IM_HEIGHT))
        self.assertTrue(array_full_equal(np.array(pillow_image), self.arr))
        self.assertEqual(pillow_image.mode, SUPPORT_MODE)
        self.assertEqual(pillow_image.width, IM_WIDTH)
        self.assertEqual(pillow_image.height, IM_HEIGHT)

    def test_pillow_to_image_and_numpy_to_image_should_same(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )

        pillow_image = PImage.fromarray(np_arr, mode=SUPPORT_MODE)
        pillow_image_arr = np.array(pillow_image)
        self.assertTrue(np.array_equal(np_arr, pillow_image_arr))

        p_src_image = mm.Image.from_pillow(pillow_image)
        N_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        self.assertTrue(np.array_equal(np_arr, N_src_image.numpy()))
        self.assertTrue(np.array_equal(p_src_image.numpy(), np_arr))

    def test_image_crop_should_success(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        dst_image = n_src_image.crop(0, 0, CROP_HEIGHT, CROP_WIDTH, mm.DeviceMode.CPU)
        self.assertEqual(dst_image.format, ImageFormat.RGB)
        self.assertEqual(dst_image.dtype, DataType.UINT8)
        self.assertEqual(dst_image.size, [CROP_WIDTH, CROP_HEIGHT])
        self.assertEqual(dst_image.device, "cpu")
        self.assertEqual(dst_image.nbytes, CROP_WIDTH * CROP_HEIGHT * THREE_CHANNEL)
        p_image = PImage.fromarray(np_arr, mode=SUPPORT_MODE)
        img1 = dst_image.numpy()
        img2 = np.array(p_image.crop((0, 0, CROP_WIDTH, CROP_HEIGHT)))
        self.assertTrue(np.array_equal(img1, img2))

    def test_image_crop_should_success_with_default_params(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        dst_image = n_src_image.crop(0, 0, CROP_HEIGHT, CROP_WIDTH)
        self.assertEqual(dst_image.format, ImageFormat.RGB)
        self.assertEqual(dst_image.dtype, DataType.UINT8)
        self.assertEqual(dst_image.size, [CROP_WIDTH, CROP_HEIGHT])
        self.assertEqual(dst_image.device, "cpu")
        self.assertEqual(dst_image.nbytes, CROP_WIDTH * CROP_HEIGHT * THREE_CHANNEL)
        p_image = PImage.fromarray(np_arr, mode=SUPPORT_MODE)
        img1 = dst_image.numpy()
        img2 = np.array(p_image.crop((0, 0, CROP_WIDTH, CROP_HEIGHT)))
        self.assertTrue(np.array_equal(img1, img2))

    def test_image_crop_should_fail_with_invalid_params(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        with self.assertRaises(AttributeError):
            dst_image = n_src_image.crop(0, 0, CROP_HEIGHT, CROP_WIDTH, 1)
        with self.assertRaises(RuntimeError):
            dst_image = n_src_image.crop(0, 0, 0, 0, mm.DeviceMode.CPU)
        with self.assertRaises(RuntimeError):
            dst_image = n_src_image.crop(
                0, 0, HEIGHT_840 + 1, WIDTH_960 + 1, mm.DeviceMode.CPU
            )
        np_arr = np.random.randint(
            0, 256, (THREE_CHANNEL, HEIGHT_840, WIDTH_960), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB_PLANAR)
        with self.assertRaises(RuntimeError):
            dst_image = n_src_image.crop(
                0, 0, CROP_HEIGHT, CROP_WIDTH, mm.DeviceMode.CPU
            )

    def test_image_resize_success_with_valid_params(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        dst_image = n_src_image.resize(
            (RESIZE_WIDTH, RESIZE_HEIGHT), mm.Interpolation.BICUBIC, mm.DeviceMode.CPU
        )
        self.assertEqual(dst_image.format, ImageFormat.RGB)
        self.assertEqual(dst_image.dtype, DataType.UINT8)
        self.assertEqual(dst_image.size, [RESIZE_WIDTH, RESIZE_HEIGHT])
        self.assertEqual(dst_image.device, "cpu")
        self.assertEqual(dst_image.nbytes, RESIZE_WIDTH * RESIZE_HEIGHT * THREE_CHANNEL)
        p_image = PImage.fromarray(np_arr, mode=SUPPORT_MODE)
        img1 = dst_image.numpy()
        img2 = np.array(p_image.resize((RESIZE_WIDTH, RESIZE_HEIGHT), PImage.BICUBIC))
        self.assertTrue(np.array_equal(img1, img2))

    def test_image_resize_success_with_default_params(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        dst_image = n_src_image.resize(
            (RESIZE_WIDTH, RESIZE_HEIGHT), mm.Interpolation.BICUBIC
        )
        self.assertEqual(dst_image.format, ImageFormat.RGB)
        self.assertEqual(dst_image.dtype, DataType.UINT8)
        self.assertEqual(dst_image.size, [RESIZE_WIDTH, RESIZE_HEIGHT])
        self.assertEqual(dst_image.device, "cpu")
        self.assertEqual(dst_image.nbytes, RESIZE_WIDTH * RESIZE_HEIGHT * THREE_CHANNEL)
        p_image = PImage.fromarray(np_arr, mode=SUPPORT_MODE)
        img1 = dst_image.numpy()
        img2 = np.array(p_image.resize((RESIZE_WIDTH, RESIZE_HEIGHT), PImage.BICUBIC))
        self.assertTrue(np.array_equal(img1, img2))

    def test_image_resize_failed_with_invalid_params(self):
        np_arr = np.random.randint(
            0, 256, (HEIGHT_840, WIDTH_960, THREE_CHANNEL), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB)
        with self.assertRaises(AttributeError):
            dst_image = n_src_image.resize(
                (RESIZE_WIDTH, RESIZE_HEIGHT), 1, mm.DeviceMode.CPU
            )
        with self.assertRaises(AttributeError):
            dst_image = n_src_image.resize(
                (RESIZE_WIDTH, RESIZE_HEIGHT), mm.Interpolation.BICUBIC, 1
            )
        with self.assertRaises(RuntimeError):
            dst_image = n_src_image.resize(
                (INVALID_WIDTH, INVALID_HEIGHT),
                mm.Interpolation.BICUBIC,
                mm.DeviceMode.CPU,
            )
        with self.assertRaises(ValueError):
            dst_image = n_src_image.resize(
                (RESIZE_WIDTH, RESIZE_WIDTH, RESIZE_WIDTH),
                mm.Interpolation.BICUBIC,
                mm.DeviceMode.CPU,
            )
        np_arr = np.random.randint(
            0, 256, (THREE_CHANNEL, HEIGHT_840, WIDTH_960), dtype=np.uint8
        )
        n_src_image = mm.Image.from_numpy(np_arr, ImageFormat.RGB_PLANAR)
        with self.assertRaises(RuntimeError):
            dst_image = n_src_image.resize(
                (RESIZE_WIDTH, RESIZE_HEIGHT),
                mm.Interpolation.BICUBIC,
                mm.DeviceMode.CPU,
            )

    def test_image_to_tensor_should_success(self):
        image = Image.from_pillow(self.pillow_image)
        tensor = image.to_tensor()
        mm_torch_tensor = tensor.torch()

        torchvision_torch_tensor = transforms.ToTensor()(self.pillow_image)
        self.assertTrue(compare_float_torch_tensors(mm_torch_tensor, torchvision_torch_tensor.unsqueeze(0)))

        tensor = image.to_tensor(TensorFormat.NCHW)
        mm_torch_tensor = tensor.torch()
        print(mm_torch_tensor.shape)
        self.assertTrue(compare_float_torch_tensors(mm_torch_tensor, torchvision_torch_tensor.unsqueeze(0)))

        tensor = image.to_tensor(TensorFormat.NHWC)
        mm_torch_tensor = tensor.torch()

        torchvision_torch_tensor = torchvision_torch_tensor.permute(1, 2, 0)
        self.assertTrue(compare_float_torch_tensors(mm_torch_tensor, torchvision_torch_tensor.unsqueeze(0)))

    def test_image_to_tensor_should_failed_with_invalid_target_format(self):
        image = Image.from_pillow(self.pillow_image)
        with self.assertRaises(RuntimeError) as context:
            tensor = image.to_tensor(TensorFormat.ND)
        expected_message = "Failed to execute 'to tensor' operator, please ensure your inputs are valid."
        self.assertEqual(str(context.exception), expected_message)


if __name__ == "__main__":
    unittest.main()
