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
Description: Python unit tests for QwenFusion Preprocess
Author: ACC SDK
Create: 2025
History: NA
"""

import sys
import ctypes
from accsdk_pytest import BaseTestCase
import acc

DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64
DEFAULT_CHANNEL = 3
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280


def create_buffer_ptr(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, channels=DEFAULT_CHANNEL):
    total_size = width * height * channels
    buf = bytearray((i % 256 for i in range(total_size)))
    return buf


class FakeArray:
    """Simulate a numpy array with __array_interface__"""

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
            "data": (addr, False),
            "shape": shape,
            "typestr": typestr,
        }


class TestQwenPreprocess(BaseTestCase):

    def test_empty_input_should_fail(self):
        with self.assertRaisesRegex(RuntimeError, "Images is empty"):
            acc.Qwen2VLProcessor.Preprocess([], MEAN, STD, DEFAULT_WIDTH, DEFAULT_HEIGHT)

    def test_single_valid_image_should_success(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), "|u1")
        image = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        tensors = acc.Qwen2VLProcessor.Preprocess([image], MEAN, STD, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.assertGreater(len(tensors), 0)
        for t in tensors:
            np_dict = t.numpy()
            self.assertIn("__array_interface__", np_dict)

    def test_multiple_valid_images_should_success(self):
        images = []
        for _ in range(4):
            buf = create_buffer_ptr()
            fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), "|u1")
            images.append(acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu"))
        tensors = acc.Qwen2VLProcessor.Preprocess(images, MEAN, STD, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.assertGreater(len(tensors), 0)
        self.assertEqual(len(tensors), len(images))

    def test_preprocess_with_invalid_mean_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), "|u1")
        img = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        with self.assertRaises(RuntimeError):
            acc.Qwen2VLProcessor.Preprocess([img], [1.0], STD, DEFAULT_WIDTH, DEFAULT_HEIGHT)

    def test_preprocess_with_invalid_std_should_fail(self):
        buf = create_buffer_ptr()
        fake_np = FakeArray(buf, (DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_CHANNEL), "|u1")
        img = acc.Image.from_numpy(fake_np, acc.ImageFormat_RGB, b"cpu")
        with self.assertRaises(RuntimeError):
            acc.Qwen2VLProcessor.Preprocess([img], MEAN, [0.1], DEFAULT_WIDTH, DEFAULT_HEIGHT)


if __name__ == "__main__":
    failed = TestQwenPreprocess.run_tests()
    sys.exit(1 if failed > 0 else 0)
