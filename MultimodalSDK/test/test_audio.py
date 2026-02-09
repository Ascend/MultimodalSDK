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
import unittest
import numpy as np

from mm import load_audio

VALID_AUDIO_PATH = "./test/assets/audios/audio_test.wav"
INVALID_AUDIO_PATH = "./test/assets/not_exist.wav"
VALID_AUDIO_DIR = "./test/assets/audios"


class TestAudio(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000

    def test_load_audio_from_single_file_should_success(self):
        os.chmod(VALID_AUDIO_PATH, 0o440)
        result = load_audio(VALID_AUDIO_PATH, self.sample_rate)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        tensor, sample_rate = result

        self.assertEqual(len(tensor.shape), 1)
        self.assertEqual(sample_rate, self.sample_rate)
        self.assertGreater(tensor.shape[0], 0)

    def test_load_audio_with_default_sr_should_success(self):
        os.chmod(VALID_AUDIO_PATH, 0o440)
        result = load_audio(VALID_AUDIO_PATH)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        tensor, sample_rate = result

        self.assertTrue(hasattr(tensor, 'shape'))
        self.assertEqual(len(tensor.shape), 1)
        self.assertGreater(tensor.shape[0], 0)
        self.assertGreater(sample_rate, 0)

    def test_load_audio_batch_from_list_should_success(self):
        os.chmod(VALID_AUDIO_PATH, 0o440)
        files = [VALID_AUDIO_PATH, VALID_AUDIO_PATH]
        results = load_audio(files, self.sample_rate)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            tensor, sr = item

            self.assertTrue(hasattr(tensor, 'shape'))
            self.assertTrue(hasattr(tensor, 'dtype'))
            self.assertEqual(len(tensor.shape), 1)
            self.assertEqual(sr, self.sample_rate)
            self.assertGreater(tensor.shape[0], 0)

    def test_load_audio_from_directory_should_success(self):
        os.chmod(VALID_AUDIO_PATH, 0o440)
        dir_path = VALID_AUDIO_DIR
        results = load_audio(dir_path, self.sample_rate)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            tensor, sr = item

            self.assertTrue(hasattr(tensor, 'shape'))
            self.assertTrue(hasattr(tensor, 'dtype'))
            self.assertEqual(len(tensor.shape), 1)
            self.assertEqual(sr, self.sample_rate)
            self.assertGreater(tensor.shape[0], 0)

    def test_load_audio_with_invalid_path_type_should_fail(self):
        with self.assertRaises(TypeError):
            load_audio(12345, self.sample_rate)

    def test_load_audio_with_sr_not_int_should_fail(self):
        with self.assertRaises(ValueError) as cm:
            load_audio(VALID_AUDIO_PATH, 44100.0)

        self.assertIn("sr must be positive int", str(cm.exception))

    def test_load_audio_with_sr_not_positive_should_fail(self):
        with self.assertRaises(ValueError):
            load_audio(VALID_AUDIO_PATH, 0)

        with self.assertRaises(ValueError):
            load_audio(VALID_AUDIO_PATH, -16000)

    def test_load_audio_from_not_exist_file_should_fail(self):
        with self.assertRaises(ValueError):
            load_audio(INVALID_AUDIO_PATH, self.sample_rate)

    def test_load_audio_from_empty_list_should_fail(self):
        with self.assertRaises(ValueError):
            load_audio([], self.sample_rate)


if __name__ == "__main__":
    unittest.main()