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
Description: python video decode test.
Author: ACC SDK
Create: 2025
History: NA
"""


import os
import sys
import ctypes
from accsdk_pytest import BaseTestCase
import acc

VALID_VIDEO_PATH = b"../../data/videos/video_1min_30fps.mp4"
VALID_VIDEO_ABSOLUTE_PATH = os.path.abspath("../../data/videos/video_1min_30fps.mp4").encode('utf-8')
VIDEO_PATH_WITHOUT_VIDEO_STREAM = b"../../data/videos/test_acc.mp4"
DEVICE_CPU = b"cpu"
DEVICE_NPU = b"npu"
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
VIDEO_DECODED_FORMAT = 12


class TestPyVideo(BaseTestCase):
    def test_video_decode_absolute_path(self):
        target_indices = {1, 3, 5}
        sample_num = 0
        os.chmod(VALID_VIDEO_ABSOLUTE_PATH, 0o440)
        frames = acc.video_decode(VALID_VIDEO_ABSOLUTE_PATH, DEVICE_CPU, target_indices, sample_num)
        self.assertEqual(len(frames), len(target_indices))
        for frame in frames:
            self.assertEqual(frame.width, EXPECTED_WIDTH)
            self.assertEqual(frame.height, EXPECTED_HEIGHT)
            self.assertEqual(frame.format, VIDEO_DECODED_FORMAT)

    def test_video_decode_success(self):
        frames = []
        target_indices = {1, 2, 5, 8}
        os.chmod(VALID_VIDEO_PATH, 0o440)
        frames = acc.video_decode(VALID_VIDEO_PATH, DEVICE_CPU, target_indices, 1)
        self.assertEqual(len(frames), 4)
        for frame in frames:
            self.assertEqual(frame.width, EXPECTED_WIDTH)
            self.assertEqual(frame.height, EXPECTED_HEIGHT)
            self.assertEqual(frame.format, VIDEO_DECODED_FORMAT)

    def test_video_decode_invalid_path(self):
        target_indices = {0, 1, 2}
        with self.assertRaises(RuntimeError):
            acc.video_decode(b"invalid_path.mp4", DEVICE_CPU, target_indices)

    def test_video_decode_no_video_stream(self):
        target_indices = {0, 1, 2}
        os.chmod(VIDEO_PATH_WITHOUT_VIDEO_STREAM, 0o440)
        with self.assertRaises(RuntimeError):
            acc.video_decode(VIDEO_PATH_WITHOUT_VIDEO_STREAM, DEVICE_CPU, target_indices)

    def test_video_decode_frames_count_equal_target_indices(self):
        target_indices = {0, 1, 2, 10, 22, 33}
        os.chmod(VALID_VIDEO_PATH, 0o440)
        frames = acc.video_decode(VALID_VIDEO_PATH, DEVICE_CPU, target_indices)
        self.assertEqual(len(frames), len(target_indices))

    def test_video_decode_with_sample_num_and_target_indices(self):
        target_indices = {0, 1, 2, 10, 22, 33}
        os.chmod(VALID_VIDEO_PATH, 0o440)
        frames = acc.video_decode(VALID_VIDEO_PATH, DEVICE_CPU, target_indices, 1200)
        self.assertEqual(len(frames), len(target_indices))

    def test_video_decode_invalid_when_both_empty(self):
        os.chmod(VALID_VIDEO_PATH, 0o440)
        with self.assertRaises(RuntimeError):
            acc.video_decode(VALID_VIDEO_PATH, DEVICE_CPU)

    def test_video_decode_sample_num_valid_target_indices_empty(self):
        target_indices = set()
        os.chmod(VALID_VIDEO_PATH, 0o440)
        frames = acc.video_decode(VALID_VIDEO_PATH, DEVICE_CPU, target_indices, 20)
        self.assertEqual(len(frames), 20)
        for frame in frames:
            self.assertEqual(frame.width, EXPECTED_WIDTH)
            self.assertEqual(frame.height, EXPECTED_HEIGHT)
            self.assertEqual(frame.format, VIDEO_DECODED_FORMAT)

    def test_video_decode_device_npu_invalid(self):
        target_indices = {0, 1, 2, 10, 22, 200}
        os.chmod(VALID_VIDEO_PATH, 0o440)
        with self.assertRaises(RuntimeError):
            acc.video_decode(VALID_VIDEO_PATH, DEVICE_NPU, target_indices, 20)

    def test_video_decode_target_indices_out_of_range_border(self):
        target_indices = {0, 1, 2, 10, 22, 1802}
        os.chmod(VALID_VIDEO_PATH, 0o440)
        with self.assertRaises(RuntimeError):
            acc.video_decode(VALID_VIDEO_PATH, DEVICE_CPU, target_indices, 20)

if __name__ == '__main__':
    failed = TestPyVideo.run_tests()
    sys.exit(1 if failed > 0 else 0)