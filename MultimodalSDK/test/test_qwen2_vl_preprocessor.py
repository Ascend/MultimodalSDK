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
import unittest
import numpy as np
from PIL import Image as PILImage
from mm import Image, ImageFormat, MultimodalQwen2VLImageProcessor
from mm.adapter.qwen2_vl_preprocessor import _check_image, ImageConstraints


class Test_Qwen2VL_Preprocess(unittest.TestCase):
    BATCH_SIZE = 8
    PATCH_SIZE = 14
    MERGE_SIZE = 2
    MIN_DIM = PATCH_SIZE * MERGE_SIZE
    MAX_DIM = 1024
    FRAME_SIZE = 32
    MIN_PIXELS = 3136
    MAX_PIXELS = 28 * 28 * 1280
    TEMPORAL_PATCH_SIZE = 2

    def random_image_array(self, min_size=None, max_size=None, channels=3):
        min_size = min_size or self.MIN_DIM
        max_size = max_size or self.MAX_DIM
        h = np.random.randint(min_size, max_size + 1)
        w = np.random.randint(min_size, max_size + 1)
        arr = np.random.randint(0, 256, (h, w, channels), dtype=np.uint8)
        return arr

    def _make_cons(self, **kwargs):
        return ImageConstraints(
            patch_size=kwargs.get("patch_size", self.PATCH_SIZE),
            merge_size=kwargs.get("merge_size", self.MERGE_SIZE),
            min_pixels=kwargs.get("min_pixels", self.MIN_PIXELS),
            max_pixels=kwargs.get("max_pixels", self.MAX_PIXELS),
            temporal_patch_size=kwargs.get("temporal_patch_size", self.TEMPORAL_PATCH_SIZE),
        )

    def random_video_list(self, num_frames=None):
        num_frames = num_frames or np.random.randint(2, 5)
        return [self.random_image_array() for _ in range(num_frames)]

    def test_valid_image(self):
        arr = self.random_image_array()
        img = Image.from_numpy(arr, ImageFormat.RGB)
        cons = self._make_cons()
        _check_image(img, cons)

    def test_size_too_large(self):
        arr = self.random_image_array(min_size=5000, max_size=5000)
        img = Image.from_numpy(arr, ImageFormat.RGB)
        cons = self._make_cons()
        with self.assertRaises(ValueError):
            _check_image(img, cons)

    def test_patch_merge_exceeds(self):
        arr = self.random_image_array(min_size=20, max_size=20)
        img = Image.from_numpy(arr, ImageFormat.RGB)
        cons = self._make_cons(patch_size=14, merge_size=2)  # patch_merge=28
        with self.assertRaises(ValueError):
            _check_image(img, cons)

    def test_max_pixels_too_small(self):
        arr = self.random_image_array()
        img = Image.from_numpy(arr, ImageFormat.RGB)
        cons = self._make_cons(max_pixels=10)
        with self.assertRaises(ValueError):
            _check_image(img, cons)

    def test_min_pixels_ge_max_pixels(self):
        arr = self.random_image_array()
        img = Image.from_numpy(arr, ImageFormat.RGB)
        cons = self._make_cons(min_pixels=1000, max_pixels=500)
        with self.assertRaises(ValueError):
            _check_image(img, cons)

    def test_temporal_patch_size_wrong(self):
        arr = self.random_image_array()
        img = Image.from_numpy(arr, ImageFormat.RGB)
        cons = self._make_cons(temporal_patch_size=1)
        with self.assertRaises(ValueError):
            _check_image(img, cons)

    def test_valid_images_single_and_batch(self):
        arr_list = [self.random_image_array() for _ in range(self.BATCH_SIZE)]
        pil_list = [PILImage.fromarray(arr) for arr in arr_list]
        image_list = [Image.from_numpy(arr, ImageFormat.RGB) for arr in arr_list]

        bf1 = MultimodalQwen2VLImageProcessor().preprocess(arr_list[0])
        self.assertIn("pixel_values", bf1.data)

        bf2 = MultimodalQwen2VLImageProcessor().preprocess(arr_list)
        self.assertIn("pixel_values", bf2.data)

        bf3 = MultimodalQwen2VLImageProcessor().preprocess(pil_list)
        self.assertIn("pixel_values", bf3.data)

        bf4 = MultimodalQwen2VLImageProcessor().preprocess(image_list)
        self.assertIn("pixel_values", bf4.data)

        bf5 = MultimodalQwen2VLImageProcessor().preprocess(image_list[0])
        self.assertIn("pixel_values", bf5.data)

    def test_valid_videos_various_formats(self):
        H, W = 64, 64
        video_frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(self.FRAME_SIZE)]
        bf1 = MultimodalQwen2VLImageProcessor().preprocess(None, videos=video_frames)
        self.assertIn("pixel_values_videos", bf1.data)

        multi_videos = [
            [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(np.random.randint(2, 5))]
            for _ in range(2)
        ]
        bf2 = MultimodalQwen2VLImageProcessor().preprocess(None, videos=multi_videos)
        self.assertIn("pixel_values_videos", bf2.data)

        video_array = np.stack(video_frames, axis=0)
        bf3 = MultimodalQwen2VLImageProcessor().preprocess(None, videos=video_array)
        self.assertIn("pixel_values_videos", bf3.data)

        multi_video_arrays = [
            np.stack([np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
                      for _ in range(np.random.randint(2, 5))], axis=0)
            for _ in range(2)
        ]
        bf4 = MultimodalQwen2VLImageProcessor().preprocess(None, videos=multi_video_arrays)
        self.assertIn("pixel_values_videos", bf4.data)

        mixed_video = [
            Image.from_numpy(np.random.randint(0, 256, (H, W, 3), dtype=np.uint8), ImageFormat.RGB),
            PILImage.fromarray(np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)),
            np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        ]
        bf5 = MultimodalQwen2VLImageProcessor().preprocess(None, videos=mixed_video)
        self.assertIn("pixel_values_videos", bf5.data)

    def test_video_and_image_only_video_used(self):
        video = [self.random_video_list()]
        arr = self.random_image_array()
        bf = MultimodalQwen2VLImageProcessor().preprocess(images=arr, videos=video)
        self.assertIn("pixel_values_videos", bf.data)
        self.assertNotIn("pixel_values", bf.data)

    def test_image_exceptions(self):
        tiny = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(tiny)

        arr = self.random_image_array(min_size=self.MIN_DIM)
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(arr, patch_size=arr.shape[0] + 1, merge_size=2)

        arr = self.random_image_array()
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(arr, patch_size=self.PATCH_SIZE, merge_size=self.MERGE_SIZE,
                                                         max_pixels=1)

        arr = self.random_image_array()
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(arr, patch_size=self.PATCH_SIZE, merge_size=self.MERGE_SIZE,
                                                         min_pixels=3136, max_pixels=3136)

        arr = self.random_image_array()
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(arr, temporal_patch_size=3)

    def test_video_frame_shape_exceptions(self):
        bad_video = [np.random.randint(0, 256, (10, 10), dtype=np.uint8)]
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[bad_video])

        bad_video2 = [np.random.randint(0, 256, (10, 10, 4), dtype=np.uint8)]
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[bad_video2])

    def test_image_type_exceptions(self):
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess("not an image")
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess([123, 456])
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[["invalid frame"]])

    def test_batched_videos_single_frame_ndarray(self):
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        bf = MultimodalQwen2VLImageProcessor().preprocess(None, videos=frame)
        self.assertIn("pixel_values_videos", bf.data)

    def test_batched_videos_list_of_frames_mixed_types(self):
        frame1 = Image.from_numpy(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8), ImageFormat.RGB)
        frame2 = PILImage.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        frame3 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        frames = [frame1, frame2, frame3]
        bf = MultimodalQwen2VLImageProcessor().preprocess(None, videos=frames)
        self.assertIn("pixel_values_videos", bf.data)

    def test_batched_videos_single_image_frame(self):
        frame = Image.from_numpy(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8), ImageFormat.RGB)
        bf = MultimodalQwen2VLImageProcessor().preprocess(None, videos=frame)
        self.assertIn("pixel_values_videos", bf.data)

    def test_batched_videos_single_pil_frame(self):
        frame = PILImage.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        bf = MultimodalQwen2VLImageProcessor().preprocess(None, videos=frame)
        self.assertIn("pixel_values_videos", bf.data)

    def test_batched_videos_with_wrong_shape_and_type(self):
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        frames_wrong_shape = [frame, np.random.randint(0, 256, (32, 32), dtype=np.uint8)]
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=frames_wrong_shape)

        frames_wrong_type = [frame, None]
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=frames_wrong_type)

    def test_image_min_max_edge_cases(self):
        arr_min = self.random_image_array(min_size=self.MIN_DIM, max_size=self.MIN_DIM)
        bf_min = MultimodalQwen2VLImageProcessor().preprocess(arr_min)
        self.assertIn("pixel_values", bf_min.data)

        arr_max = self.random_image_array(min_size=self.MAX_DIM, max_size=self.MAX_DIM)
        bf_max = MultimodalQwen2VLImageProcessor().preprocess(arr_max)
        self.assertIn("pixel_values", bf_max.data)

    def test_image_non_square_dimensions(self):
        h = self.MIN_DIM
        w = self.MIN_DIM * 2
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        bf = MultimodalQwen2VLImageProcessor().preprocess(arr)
        self.assertIn("pixel_values", bf.data)

    def test_video_single_frame(self):
        frame = self.random_image_array()
        bf = MultimodalQwen2VLImageProcessor().preprocess(None, videos=[frame])
        self.assertIn("pixel_values_videos", bf.data)

    def test_video_frames_invalid_channels_raise(self):
        gray_frame = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        rgba_frame = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)

        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[gray_frame])

        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[rgba_frame])

    def test_none_data_should_pass(self):
        MultimodalQwen2VLImageProcessor().preprocess(None, videos=None)

    def test_image_empty_list_should_pass(self):
        MultimodalQwen2VLImageProcessor().preprocess(None, videos=None)

    def test_video_empty_list_should_fail(self):
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[])

        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[[], []])

        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=[[[]], []])

    def test_min_max_pixels_edge_case(self):
        arr = self.random_image_array(min_size=50, max_size=50)
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(arr, min_pixels=2500, max_pixels=2500)

    def test_temporal_patch_size_invalid(self):
        arr = self.random_video_list(num_frames=2)
        with self.assertRaises(ValueError):
            MultimodalQwen2VLImageProcessor().preprocess(arr, temporal_patch_size=5)

    def test_video_frames_with_invalid_types(self):
        frames = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
            "not an image",
            None
        ]
        with self.assertRaises(TypeError):
            MultimodalQwen2VLImageProcessor().preprocess(None, videos=frames)
            MultimodalQwen2VLImageProcessor().preprocess([[], [[], []], [], []], videos=[[], [], [], [[], []]])

    def test_images_and_videos_mixed_types_video_used(self):
        images = [
            np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
            PILImage.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        ]
        videos = [self.random_video_list()]
        bf = MultimodalQwen2VLImageProcessor().preprocess(images=images, videos=videos)
        self.assertIn("pixel_values_videos", bf.data)
        self.assertNotIn("pixel_values", bf.data)

    def test_invalid_mean_std_value(self):
        videos = [self.random_video_list()]
        invalid_mean = [-1, [0.1, 0.1], [1.1, 1.1, 1.1], 1.1, [1.1], videos]
        mean_error = [ValueError, TypeError, ValueError, ValueError, TypeError, TypeError]
        processor = MultimodalQwen2VLImageProcessor()
        for val, err in zip(invalid_mean, mean_error):
            with self.assertRaises(err):
                processor.preprocess(images=[], videos=videos, image_mean=val)
        invalid_std = [
            -1,  # negative value
            [0.0, 0.0, 0.0],  # contains zero
            [3.5e38, 3.5e38, 3.5e38],  # exceeds max float
            0.0,  # float = 0
            [1.0, -2.0, 1.0],  # contains negative
            "std",  # wrong type
        ]
        std_error = [
            ValueError,
            ValueError,
            TypeError,
            ValueError,
            ValueError,
            TypeError,
        ]
        for val, err in zip(invalid_std, std_error):
            with self.assertRaises(err):
                processor.preprocess(images=[], videos=videos, image_std=val)



if __name__ == "__main__":
    unittest.main()