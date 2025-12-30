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
from torchvision.transforms import functional as F

import accdata.backend as _b
from accdata import ops
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensor

from ut.utils import RandomDataSource, TorchOpTransforms


@pytest.mark.smoke
class NormalizeTest(unittest.TestCase):
    def setUp(self):
        _b.SetLogLevel("error")
        self.mean = 0.485
        self.std = 0.229
        self.scale = 0.5

    def accdata_norm(self, batch_size, thread_cnt, queue_depth, test_data):
        pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
        with pipe:
            data_source = ops.external_source("Source")
            norm = ops.normalize(data_source.output, mean=[self.mean, self.mean, self.mean],
                                 std=[self.std, self.std, self.std],
                                 scale=self.scale)
        pipe.build([data_source.spec, norm.spec], [norm.output])
        for source in test_data:
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            self.run_pipeline_with_accuracy_check(feed_data, pipe, source)

    def accdata_norm_with_default_inputs(self, batch_size, thread_cnt, queue_depth, test_data):
        # default scale is 1.0
        self.scale = 1.0
        pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
        with pipe:
            data_source = ops.external_source("Source")
            norm = ops.normalize(data_source.output, mean=[self.mean, self.mean, self.mean],
                                 std=[self.std, self.std, self.std])
        pipe.build([data_source.spec, norm.spec], [norm.output])
        for source in test_data:
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            self.run_pipeline_with_accuracy_check(feed_data, pipe, source)

        self.scale = 0.5

    def run_pipeline_with_accuracy_check(self, feed_data, pipe, source):
        accdata_outputs = pipe.run(**feed_data)
        self.assertIsNotNone(accdata_outputs)
        accdata_torch_outputs = to_torch_tensor(accdata_outputs[0][0])
        expect = TorchOpTransforms.normalize(source.tensor.clone(), self)
        self.assertTrue(
            TorchOpTransforms.compare_tensors(accdata_torch_outputs,
                                              expect))

    def test_norm_with_error_mean_size(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.normalize(data_source.output, mean=[self.mean, self.mean],
                              std=[self.std, self.std, self.std])
            self.assertIn("Expect '3' elements, received '2'", str(context.exception))

    def test_norm_with_error_mean_type(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(TypeError) as context:
                ops.normalize(data_source.output, mean=1,
                              std=[self.std, self.std, self.std])
            self.assertIn("Expected 'mean' argument of list of type", str(context.exception))

    def test_norm_with_error_mean_list_type(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(TypeError) as context:
                ops.normalize(data_source.output, mean=self.mean,
                              std=[self.std, self.std, self.std])
            self.assertIn("Expected 'mean' argument of list of type", str(context.exception))

    def test_norm_with_error_layout(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image = ops.external_source("Image")
            norm = ops.normalize(image.output, mean=[self.mean, self.mean, self.mean],
                                 std=[self.std, self.std, self.std])
        pipe.build([image.spec, norm.spec], [norm.output])

        # Run pipeline
        tensor = torch.rand((256, 640, 3, 340), dtype=torch.float)
        with self.assertRaises(TypeError) as context:
            inputs = {image.output.name: to_accdata_tensorlist([tensor.clone()])}
            self.pipe.run(**inputs)
        self.assertIn("Only NCHW and NHWC supported.", str(context.exception))

    def test_norm_with_different_type_tensor(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image = ops.external_source("Image")
            norm = ops.normalize(image.output, mean=[self.mean, self.mean, self.mean],
                                 std=[self.std, self.std, self.std])
        pipe.build([image.spec, norm.spec], [norm.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.randint(0, 255, (1, 3, 1080, 1920), dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as context:
            inputs = {image.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_norm_with_different_layout_or_shape_tensor(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image = ops.external_source("Image")
            norm = ops.normalize(image.output, mean=[self.mean, self.mean, self.mean],
                                 std=[self.std, self.std, self.std])
        pipe.build([image.spec, norm.spec], [norm.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 1080, 1920, 3), dtype=torch.float)
        with self.assertRaises(RuntimeError) as context:
            inputs = {image.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_norm_with_two_std(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            image = ops.external_source("Image")
            with self.assertRaises(ValueError) as context:
                norm = ops.normalize(image.output, mean=[self.mean, self.mean, self.mean],
                                     std=[self.std, self.std])
                pipe.build([image.spec, norm.spec], [norm.output])

                # Run pipeline
                tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
                tensor1 = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
                inputs = {image.output.name: to_accdata_tensorlist([tensor, tensor1])}
                pipe.run(**inputs)
        self.assertIn("Expect '3' elements, received '2'", str(context.exception))

norm_suits = NormalizeTest()
norm_suits.setUp()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("queue_depth", [4])
@pytest.mark.parametrize("data_source", [RandomDataSource.data_float_nhwc, RandomDataSource.data_float_nchw])
@pytest.mark.smoke
def test_normalize_success(batch_size, thread_cnt, queue_depth, data_source):
    norm_suits.accdata_norm(batch_size, thread_cnt, queue_depth, data_source)
    norm_suits.accdata_norm_with_default_inputs(batch_size, thread_cnt, queue_depth, data_source)


@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("data_source",
                         [RandomDataSource.data_float_nhwc_random_shape, RandomDataSource.data_float_nchw_random_shape])
@pytest.mark.slow
def test_normalize_precision(thread_cnt, data_source):
    RandomDataSource.create_random_data_source_slow()
    norm_suits.accdata_norm(8, thread_cnt, 4, data_source)
    norm_suits.accdata_norm_with_default_inputs(8, thread_cnt, 4, data_source)


if __name__ == '__main__':
    unittest.main()
