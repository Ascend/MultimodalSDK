#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""
Description: python device test.
Author: ACC SDK
Create: 2025
History: NA
"""
import os
import sys
import ctypes
from accsdk_pytest import BaseTestCase
import acc

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_CHANNEL = 3
CROP_HEIGHT = 10
CROP_WIDTH = 11
RESIZE_HEIGHT = 40
RESIZE_WIDTH = 50
INVALID_HEIGHT = 8200
INVALID_WIDTH = 8200


def create_buffer_ptr(width: int = DEFAULT_WIDTH,
                      height: int = DEFAULT_HEIGHT,
                      channels: int = DEFAULT_CHANNEL):
    total_size = width * height * channels
    buf = bytearray((i % 256 for i in range(total_size)))
    return buf


class FakeArray:
    def __init__(self, buf, shape, typestr):
        if isinstance(buf, (bytes, bytearray, memoryview)):
            addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        elif isinstance(buf, int):
            addr = buf
        else:
            addr = 0
        self._buf = buf
        self.__array_interface__ = {
            "version": 3,
            "data": (addr, False),  # start addr and read-only sign
            "shape": shape,
            "typestr": typestr,
        }


class TestPyImage(BaseTestCase):
    valid_path = None
    invalid_path = None

    @classmethod
    def setUpClass(cls):
        cls.valid_path = "../../image/assets/dog_1920_1080.jpg"
        cls.invalid_path = "../../image/assets/dog_1920_1080.png"

    def test_create_empty_image_should_fail(self):
        is_failed = False
        try:
            image = acc.Image()
        except Exception as e:
            is_failed = True
        self.assertTrue(is_failed)

    def test_create_image_from_invalid_path_should_fail(self):
        is_failed = False
        try:
            os.chmod(self.invalid_path, 0o640)
            image = acc.Image(self.invalid_path.encode("utf-8"), b"cpu")
        except Exception as e:
            is_failed = True
        self.assertTrue(is_failed)

    def test_create_image_from_valid_path_with_wrong_permission_should_fail(self):
        is_failed = False
        try:
            os.chmod(self.valid_path, 0o644)
            image = acc.Image(self.valid_path.encode("utf-8"), b"cpu")
        except Exception as e:
            is_failed = True
        self.assertTrue(is_failed)

    def test_create_image_from_valid_path_with_should_success(self):
        is_failed = False
        try:
            os.chmod(self.valid_path, 0o640)
            image = acc.Image(self.valid_path.encode("utf-8"), b"cpu")
            self.assertEqual(list(image.size), [1920, 1080])
            self.assertEqual(image.height, 1080)
            self.assertEqual(image.width, 1920)
            self.assertEqual(image.format, 12)
            self.assertEqual(image.dtype, 4)
            self.assertEqual(image.device, b"cpu")
        except Exception as e:
            self.fail(f"Expected no exception, but got {e}")

    def test_create_image_from_clone_should_success(self):
        is_failed = False
        try:
            os.chmod(self.valid_path, 0o640)
            image = acc.Image(self.valid_path.encode("utf-8"), b"cpu")
            self.assertEqual(list(image.size), [1920, 1080])
            self.assertEqual(image.height, 1080)
            self.assertEqual(image.width, 1920)
            self.assertEqual(image.format, 12)
            self.assertEqual(image.dtype, 4)
            self.assertEqual(image.device, b"cpu")
            image_clone = image.clone()
            self.assertEqual(list(image_clone.size), [1920, 1080])
            self.assertEqual(image_clone.height, 1080)
            self.assertEqual(image_clone.width, 1920)
            self.assertEqual(image_clone.format, 12)
            self.assertEqual(image_clone.dtype, 4)
            self.assertEqual(image_clone.device, b"cpu")
        except Exception as e:
            self.fail(f"Expected no exception, but got {e}")

    def test_from_numpy_creates_RGB_image_correctly(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), '|u1')
        image = acc.Image.from_numpy(fake_np, 12, b"cpu")
        self.assertEqual(list(image.size), [DEFAULT_WIDTH, DEFAULT_HEIGHT])
        self.assertEqual(image.height, DEFAULT_HEIGHT)
        self.assertEqual(image.width, DEFAULT_WIDTH)
        self.assertEqual(image.format, 12)
        self.assertEqual(image.dtype, 4)
        self.assertEqual(image.device, b"cpu")

    def test_from_numpy_creates_BGR_image_correctly(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), '|u1')
        image = acc.Image.from_numpy(fake_np, 13, b"cpu")
        self.assertEqual(list(image.size), [DEFAULT_WIDTH, DEFAULT_HEIGHT])
        self.assertEqual(image.height, DEFAULT_HEIGHT)
        self.assertEqual(image.width, DEFAULT_WIDTH)
        self.assertEqual(image.format, 13)
        self.assertEqual(image.dtype, 4)
        self.assertEqual(image.device, b"cpu")

    def test_from_numpy_creates_BGR_PLANAR_image_correctly(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH), '|u1')
        image = acc.Image.from_numpy(fake_np, 70, b"cpu")
        self.assertEqual(list(image.size), [DEFAULT_WIDTH, DEFAULT_HEIGHT])
        self.assertEqual(image.height, DEFAULT_HEIGHT)
        self.assertEqual(image.width, DEFAULT_WIDTH)
        self.assertEqual(image.format, 70)
        self.assertEqual(image.dtype, 4)
        self.assertEqual(image.device, b"cpu")

    def test_from_numpy_creates_RGB_PLANAR_image_correctly(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH), '|u1')
        image = acc.Image.from_numpy(fake_np, 69, b"cpu")
        self.assertEqual(list(image.size), [DEFAULT_WIDTH, DEFAULT_HEIGHT])
        self.assertEqual(image.height, DEFAULT_HEIGHT)
        self.assertEqual(image.width, DEFAULT_WIDTH)
        self.assertEqual(image.format, 69)
        self.assertEqual(image.dtype, 4)
        self.assertEqual(image.device, b"cpu")

    def test_from_numpy_creates_image_with_wrong_dtype_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_WIDTH, DEFAULT_HEIGHT), '|f4')
        with self.assertRaisesRegex(RuntimeError, "data type should be uint8"):
            acc.Image.from_numpy(fake_np, 69, b"cpu")

    def test_from_numpy_creates_image_with_wrong_shape_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_WIDTH, DEFAULT_HEIGHT, 1), '|u1')
        with self.assertRaisesRegex(RuntimeError, "shape should be 3D"):
            acc.Image.from_numpy(fake_np, 69, b"cpu")

    def test_from_numpy_creates_image_with_wrong_format_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_WIDTH, DEFAULT_HEIGHT), '|u1')
        with self.assertRaisesRegex(RuntimeError, "unsupported image format"):
            acc.Image.from_numpy(fake_np, -1, b"cpu")

    def test_from_numpy_creates_BGR_PLANAR_image_with_wrong_format_and_channel_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CHANNEL), '|u1')
        with self.assertRaisesRegex(RuntimeError, r"expect shape \[3, H, W\]"):
            acc.Image.from_numpy(fake_np, 70, b"cpu")

    def test_from_numpy_creates_RGB_PLANAR_image_with_wrong_format_and_channel_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CHANNEL), '|u1')
        with self.assertRaisesRegex(RuntimeError, r"expect shape \[3, H, W\]"):
            acc.Image.from_numpy(fake_np, 69, b"cpu")

    def test_from_numpy_creates_RGB_image_with_wrong_format_and_channel_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_WIDTH, DEFAULT_HEIGHT), '|u1')
        with self.assertRaisesRegex(RuntimeError, r"expect shape \[H, W, 3\]"):
            acc.Image.from_numpy(fake_np, 12, b"cpu")

    def test_from_numpy_creates_BGR_image_with_wrong_format_and_channel_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_WIDTH, DEFAULT_HEIGHT), '|u1')
        with self.assertRaisesRegex(RuntimeError, r"expect shape \[H, W, 3\]"):
            acc.Image.from_numpy(fake_np, 13, b"cpu")

    def test_numpy_returns_consistent_data(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), '|u1')
        image = acc.Image.from_numpy(fake_np, 12, b"cpu")
        np_dict = image.numpy()
        self.assertIn("__array_interface__", np_dict)
        arr_info = np_dict["__array_interface__"]
        self.assertEqual(arr_info["shape"], (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL))
        out_ptr, _readonly = arr_info["data"]
        out_size = DEFAULT_WIDTH * DEFAULT_HEIGHT * DEFAULT_CHANNEL
        out_buf = (ctypes.c_ubyte * out_size).from_address(out_ptr)
        out_bytes = bytes(out_buf)
        self.assertEqual(out_bytes, bytes(buf), "Data mismatch between input and output!")

    def test_from_numpy_null_pyobject_should_fail(self):
        with self.assertRaisesRegex(RuntimeError,
                                    "The python numpy ndarray does not have the __array_interface__ dictionary"):
            acc.Image.from_numpy(None, 12, b"cpu")

    def test_from_numpy_no_array_interface_should_fail(self):
        class NoInterface:
            pass

        with self.assertRaisesRegex(RuntimeError,
                                    "The python numpy ndarray does not have the __array_interface__ dictionary"):
            acc.Image.from_numpy(NoInterface(), 12, b"cpu")

    def test_from_numpy_invalid_data_field_should_fail(self):
        bad = FakeArray(None, (10, 10), '|u1')
        bad.__array_interface__["data"] = None
        with self.assertRaisesRegex(RuntimeError, "Invalid data field"):
            acc.Image.from_numpy(bad, 12, b"cpu")

    def test_from_numpy_invalid_shape_field_should_fail(self):
        bad = FakeArray(1234, None, '|u1')
        with self.assertRaisesRegex(RuntimeError, "Invalid shape field"):
            acc.Image.from_numpy(bad, 12, b"cpu")

    def test_from_numpy_invalid_dimension_in_shape_should_fail(self):
        bad = FakeArray(1234, ("abc",), '|u1')
        with self.assertRaisesRegex(RuntimeError, "Invalid dimension"):
            acc.Image.from_numpy(bad, 12, b"cpu")

    def test_from_numpy_invalid_typestr_should_fail(self):
        bad = FakeArray(1234, (10, 10), None)
        with self.assertRaisesRegex(RuntimeError, "Invalid typestr field"):
            acc.Image.from_numpy(bad, 12, b"cpu")

    def test_from_numpy_encode_typestr_fail_should_fail(self):
        bad = FakeArray(1234, (10, 10), 123)
        with self.assertRaises(RuntimeError):
            acc.Image.from_numpy(bad, 12, b"cpu")

    def test_from_numpy_typestr_not_in_map_should_fail(self):
        bad = FakeArray(1234, (10, 10), '|u8')
        with self.assertRaisesRegex(RuntimeError, "Unsupported python numpy ndarray datatype"):
            acc.Image.from_numpy(bad, 12, b"cpu")

    def test_from_numpy_with_wrong_device(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), '|u1')
        with self.assertRaisesRegex(RuntimeError, "device must be 'cpu'"):
            acc.Image.from_numpy(fake_np, 12, b"npu")

    def test_crop_success_with_valid_params(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(
            buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), "|u1"
        )
        src_image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        dst_image = src_image.crop(0, 0, CROP_HEIGHT, CROP_WIDTH, acc.DeviceMode_CPU)
        self.assertEqual(dst_image.format, acc.ImageFormat_RGB)
        self.assertEqual(dst_image.dtype, acc.DataType_UINT8)
        self.assertEqual(list(dst_image.size), [CROP_WIDTH, CROP_HEIGHT])
        self.assertEqual(dst_image.device, b"cpu")
        self.assertEqual(dst_image.nbytes, CROP_WIDTH * CROP_HEIGHT * DEFAULT_CHANNEL)
        np_dict = dst_image.numpy()
        self.assertIn("__array_interface__", np_dict)
        arr_info = np_dict["__array_interface__"]
        self.assertEqual(arr_info["shape"], (CROP_HEIGHT, CROP_WIDTH, DEFAULT_CHANNEL))

    def test_crop_failed_with_invalid_params(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(
            buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), "|u1"
        )
        src_image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        with self.assertRaises(RuntimeError):
            dst_image = src_image.crop(0, 0, CROP_HEIGHT, CROP_WIDTH, 1)
        with self.assertRaises(RuntimeError):
            dst_image = src_image.crop(0, 0, 0, 0, acc.DeviceMode_CPU)
        with self.assertRaises(RuntimeError):
            dst_image = src_image.crop(
                0, 0, DEFAULT_HEIGHT + 1, DEFAULT_WIDTH + 1, acc.DeviceMode_CPU
            )
        fake_np = FakeArray(
            buf, (DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH), "|u1"
        )
        src_image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB_PLANAR, b"cpu")
        with self.assertRaises(RuntimeError):
            dst_image = src_image.crop(
                0, 0, CROP_HEIGHT, CROP_WIDTH, acc.DeviceMode_CPU
            )

    def test_resize_success_with_valid_params(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), '|u1')
        src_image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        dst_image = src_image.resize(RESIZE_WIDTH, RESIZE_HEIGHT, acc.Interpolation_BICUBIC, acc.DeviceMode_CPU)
        self.assertEqual(dst_image.format, acc.ImageFormat_RGB)
        self.assertEqual(dst_image.dtype, acc.DataType_UINT8)
        self.assertEqual(list(dst_image.size), [RESIZE_WIDTH, RESIZE_HEIGHT])
        self.assertEqual(dst_image.device, b'cpu')
        self.assertEqual(dst_image.nbytes, RESIZE_WIDTH * RESIZE_HEIGHT * DEFAULT_CHANNEL)
        np_dict = dst_image.numpy()
        self.assertIn("__array_interface__", np_dict)
        arr_info = np_dict["__array_interface__"]
        self.assertEqual(arr_info["shape"], (RESIZE_HEIGHT, RESIZE_WIDTH, DEFAULT_CHANNEL))

    def test_resize_failed_with_invalid_params(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), '|u1')
        src_image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        with self.assertRaises(RuntimeError):
            dst_image = src_image.resize(RESIZE_WIDTH, RESIZE_HEIGHT, 1, acc.DeviceMode_CPU)
        with self.assertRaises(RuntimeError):
            dst_image = src_image.resize(RESIZE_WIDTH, RESIZE_HEIGHT, acc.Interpolation_BICUBIC, 1)
        with self.assertRaises(RuntimeError):
            dst_image = src_image.resize(INVALID_WIDTH, INVALID_HEIGHT, acc.Interpolation_BICUBIC, acc.DeviceMode_CPU)
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH), '|u1')
        src_image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB_PLANAR, b"cpu")
        with self.assertRaises(RuntimeError):
            dst_image = src_image.resize(RESIZE_WIDTH, RESIZE_HEIGHT, acc.Interpolation_BICUBIC, acc.DeviceMode_CPU)

    def test_to_tensor_failed_with_invalid_input(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH), '|u1')
        image = acc.Image.from_numpy(fake_np, 69, b"cpu")

        with self.assertRaises(RuntimeError):
            image.to_tensor(acc.TensorFormat_ND, acc.DeviceMode_CPU)


if __name__ == '__main__':
    failed = TestPyImage.run_tests()
    sys.exit(1 if failed > 0 else 0)
