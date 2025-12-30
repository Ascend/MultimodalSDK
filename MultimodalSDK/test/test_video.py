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
import ctypes
import os
import sys
import tempfile
import unittest
from mm import video_decode, Image
import mm

INVALID_VIDEO_PATH = "invalid_path.mp4"
VIDEO_PATH_WITHOUT_VIDEO_STREAM = "./test/assets/test_aac.mp4"
DEVICE_CPU = "cpu"
DEVICE_NPU = "npu"


class TestVideo(unittest.TestCase):

    def test_video_decode_invalid_path(self):
        target_indices = {0, 1, 2, 10, 22, 33}
        with self.assertRaises(RuntimeError):
            frames = video_decode(INVALID_VIDEO_PATH, DEVICE_CPU, target_indices)

    def test_video_decode_no_video_stream(self):
        target_indices = {0, 1, 2}
        os.chmod(VIDEO_PATH_WITHOUT_VIDEO_STREAM, 0o440)
        with self.assertRaises(RuntimeError):
            video_decode(VIDEO_PATH_WITHOUT_VIDEO_STREAM, DEVICE_CPU, target_indices)


if __name__ == '__main__':
    unittest.main()

