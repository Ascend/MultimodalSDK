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

import unittest
import pytest
import torch
import numpy as np

import accdata.backend as _b
import accdata.types as _t
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist
from accdata.plugin.pytorch import to_torch_tensorlist
from accdata import ops

from ut.utils import RandomDataSource, TorchOpTransforms


target_size_1024 = (1024, 1024)


@pytest.mark.smoke
class ResizeCropTest(unittest.TestCase):
    def setUp(self):
        pass

    def resize_crop_normal(self, test_data, batch_size=10, thread_cnt=8, queue_depth=10, interpolation_mode="bicubic"):
        for params in test_data:
            resize_size = params.get_resize_pos()
            self.target_size = params.target_size
            if not TorchOpTransforms.check_image_size(resize_size) or not\
                TorchOpTransforms.check_image_size(self.target_size):
                continue
            self.interpolation_mode = interpolation_mode
            pipe = Pipeline(batch_size, queue_depth, thread_cnt)
            with pipe:
                image_data = ops.external_source("images")
                resize_crop = ops.resize_crop(image_data.output, resize_size, params.target_size,
                                              interpolation_mode=interpolation_mode)
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([params.tensor])}
            outputs = pipe.run(**input_data)
            self.assertIsNotNone(outputs)
            accdata_ret = to_torch_tensorlist(outputs[0])

            torch_ret = TorchOpTransforms.resize_crop(params.tensor, self)
            self.assertTrue(TorchOpTransforms.compare_tensors(torch_ret, accdata_ret[0]))

    def resize_crop_with_boundary_params(self, test_data, resize_size, crop_size, interpolation_mode):
        for params in test_data:
            pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
            self.target_size = crop_size
            self.interpolation_mode = interpolation_mode
            with pipe:
                image_data = ops.external_source("images")
                resize_crop = ops.resize_crop(image_data.output, resize=resize_size, crop=crop_size,
                                              interpolation_mode=interpolation_mode)
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([params.tensor])}
            outputs = pipe.run(**input_data)
            self.assertIsNotNone(outputs)
            accdata_ret = to_torch_tensorlist(outputs[0])

            torch_ret = TorchOpTransforms.resize_crop(params.tensor, self, resize_pos=resize_size)
            self.assertTrue(TorchOpTransforms.compare_tensors(torch_ret, accdata_ret[0]))

    def test_resize_crop_with_input_uint8(self):
        params = RandomDataSource.data_uint8_nchw[0]
        resize_size = params.get_resize_pos()
        self.target_size = params.target_size
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, params.target_size,
                                            interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([params.tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_resize_crop_with_input_nhwc(self):
        params = RandomDataSource.data_float_nhwc[0]
        resize_size = params.get_resize_pos()
        self.target_size = params.target_size
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, params.target_size,
                                            interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([params.tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_resize_crop_invalid_interpolation(self):  # 测试非法插值模式输入
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, interpolation_mode="anything")
            self.assertIn("received 'anything'", str(context.exception))

    def test_resize_crop_invalid_roundmode(self):  # 测试非法舍入模式输入
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, round_mode="anything")
            self.assertIn("received 'anything'", str(context.exception))

    def test_resize_crop_resize_none(self):    # 测试resize未输入情况
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, None)
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_resize_crop_crop_height_greater_than_resize_height(self): # 测试crop的height比resize的height大的情况
        _crop = (2000, 1024)
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, crop=_crop, interpolation_mode="bilinear")
            self.assertIn("crop height size cannot greater than resize height size", str(context.exception))

    def test_resize_crop_crop_width_greater_than_resize_width(self): # 测试crop的weight比resize的weight大的情况
        _crop = (1024, 2000)
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, crop=_crop, interpolation_mode="bilinear")
            self.assertIn("crop width size cannot greater than resize width size", str(context.exception))

    def test_resize_crop_invalid_input(self): # 测试input不为DataNode_的情况
        _input = {"bad_key": 1}
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            with self.assertRaises(TypeError) as context:
                ops.resize_crop(_input, target_size_1024, interpolation_mode="bilinear")
            self.assertIn(f"Expected inputs of type 'DataNode'. "
                          f"Received input of type '{type(_input)}'.", str(context.exception))

    def test_resize_crop_none_input(self): # 测试input不为DataNode_的情况
        _input = None
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            with self.assertRaises(TypeError) as context:
                ops.resize_crop(_input, target_size_1024, interpolation_mode="bilinear")
            self.assertIn(f"Expected inputs of type 'DataNode'. "
                          f"Received input of type '{type(_input)}'.", str(context.exception))

    def test_resize_crop_with_invalid_resize_type(self): # 测试resize的类型不合法的问题
        _resize = {"bad_key": 1}
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, _resize, interpolation_mode="bilinear")
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_resize_crop_with_invalid_resize_elements(self): # 测试resize的参数不合法的问题
        _resize = (1024, "1024")
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, _resize, interpolation_mode="bilinear")
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_resize_crop_with_invalid_resize_size(self): # 测试resize的长度不为2的问题
        _resize = (1024, 1024, 1024)
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, _resize, interpolation_mode="bilinear")
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_resize_crop_with_invalid_crop_type(self): # 测试crop的类型不合法的问题
        _crop = {"bad_key": 1}
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, crop=_crop, interpolation_mode="bilinear")
            self.assertIn("Invalid crop input!", str(context.exception))

    def test_resize_crop_with_invalid_crop_elements(self): # 测试crop的参数不合法的问题
        _crop = (1024, "1024")
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, crop=_crop, interpolation_mode="bilinear")
            self.assertIn("Invalid crop input!", str(context.exception))

    def test_resize_crop_with_invalid_crop_size(self): # 测试crop的长度不为2的问题
        _crop = (1024, 1024, 1024)
        self.pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with self.pipe:
            image_data = ops.external_source("images")
            with self.assertRaises(ValueError) as context:
                ops.resize_crop(image_data.output, target_size_1024, crop=_crop, interpolation_mode="bilinear")
            self.assertIn("Invalid crop input!", str(context.exception))


    def test_resize_height_less_than_10(self):
        resize_size = (5, 1024)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_resize_height_larger_than_8192(self):
        resize_size = (8193, 1024)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_resize_width_less_than_10(self):
        resize_size = (1024, 5)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_resize_width_larger_than_8192(self):
        resize_size = (1024, 8193)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_crop_height_less_than_zero(self):
        resize_size = RandomDataSource.data_float_nchw[0].get_resize_pos()
        crop_size = (-10, 1024)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_crop_width_less_than_zero(self):
        resize_size = RandomDataSource.data_float_nchw[0].get_resize_pos()
        crop_size = (1024, -10)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_crop_height_equals_to_zero(self):
        resize_size = RandomDataSource.data_float_nchw[0].get_resize_pos()
        crop_size = (0, 1024)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_crop_width_equals_to_zero(self):
        resize_size = RandomDataSource.data_float_nchw[0].get_resize_pos()
        crop_size = (1024, 0)
        pipe = Pipeline(batch_size=10, num_threads=8, queue_depth=10)
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        with self.assertRaises(RuntimeError) as context:
            pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
            input_data = {image_data.output.name: to_accdata_tensorlist([RandomDataSource.data_float_nchw[0].tensor])}
            pipe.run(**input_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_resize_crop_with_different_type_tensor(self):
        resize_size = (10, 8192)
        crop_size = (10, 8192)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.randint(0, 255, (1, 3, 1080, 1920), dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as context:
            inputs = {image_data.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_resize_crop_with_different_layout_or_shape_tensor(self):
        resize_size = (10, 8192)
        crop_size = (10, 8192)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 1080, 1920, 3), dtype=torch.float)
        with self.assertRaises(RuntimeError) as context:
            inputs = {image_data.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_resize_crop_with_input_exceed_output(self):
        resize_size = (10, 8192)
        crop_size = (10, 8192)
        pipe = Pipeline(batch_size=1, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image_data = ops.external_source("images")
            resize_crop = ops.resize_crop(image_data.output, resize_size, crop=crop_size, interpolation_mode="bilinear")
        pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        with self.assertRaises(RuntimeError) as context:
            inputs = {image_data.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

resize_crop_test_suits = ResizeCropTest()
resize_crop_test_suits.setUp()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("queue_depth", [4])
@pytest.mark.parametrize("interpolation_mode", ["bilinear", "bicubic"])
@pytest.mark.smoke
def test_resize_crop_normal(batch_size, thread_cnt, queue_depth, interpolation_mode):
    resize_crop_test_suits.resize_crop_normal(RandomDataSource.data_float_nchw,
                                            batch_size, thread_cnt, queue_depth, interpolation_mode)


@pytest.mark.parametrize("resize_height", [20, 8192])
@pytest.mark.parametrize("resize_width", [20, 8192])
@pytest.mark.parametrize("crop_size_less", [0, 10])
@pytest.mark.parametrize("interpolation_mode", ["bilinear", "bicubic"])
@pytest.mark.smoke
def test_resize_crop_boundary_params(resize_height, resize_width, crop_size_less, interpolation_mode):
    resize_size = [resize_height, resize_width]
    crop_size = [resize_height - crop_size_less, resize_width - crop_size_less]
    resize_crop_test_suits.resize_crop_with_boundary_params(RandomDataSource.data_float_nchw,
                                                            resize_size, crop_size, interpolation_mode)


@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("interpolation_mode", ["bilinear", "bicubic"])
@pytest.mark.slow
def test_resize_crop_precision(thread_cnt, interpolation_mode):
    RandomDataSource.create_random_data_source_slow()
    resize_crop_test_suits.resize_crop_normal(RandomDataSource.data_float_nchw_random_shape,
                                            8, thread_cnt, 4, interpolation_mode)


@pytest.mark.parametrize("resize_height", [20, 8192])
@pytest.mark.parametrize("resize_width", [20, 8192])
@pytest.mark.parametrize("crop_size_less", [0, 10])
@pytest.mark.parametrize("interpolation_mode", ["bilinear", "bicubic"])
@pytest.mark.slow
def test_resize_crop_boundary_params_precision(resize_height, resize_width, crop_size_less, interpolation_mode):
    resize_size = [resize_height, resize_width]
    crop_size = [resize_height - crop_size_less, resize_width - crop_size_less]
    RandomDataSource.create_random_data_source_slow()
    resize_crop_test_suits.resize_crop_with_boundary_params(RandomDataSource.data_float_nchw_random_shape,
                                                            resize_size, crop_size, interpolation_mode)


if __name__ == "__main__":
    unittest.main()