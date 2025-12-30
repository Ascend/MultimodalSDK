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

import accdata.backend as _b
import accdata.types as _t
import accdata.ops as ops
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensor
from accdata.types import TensorLayout

from ut.utils import RandomDataSource, TorchOpTransforms


@pytest.mark.smoke
@pytest.mark.to_tensor
class ToTensorTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_to_tensor_with_wrong_layout(self):
        """测试多线程运行to_tensor算子，使用不支持的layout"""
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            target = ops.to_tensor(data_source.output)
        pipe.build([data_source.spec, target.spec], [target.output])

        tensor = torch.randint(0, 255, (1, 640, 3, 340), dtype=torch.uint8)
        with self.assertRaises(TypeError) as context:
            feed_data = {data_source.output.name: to_accdata_tensorlist([tensor.clone()])}
            pipe.run(**feed_data)
        self.assertIn("Only NCHW and NHWC supported.", str(context.exception))

    def test_to_tensor_with_different_type_tensor(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            data_source = ops.external_source("Source")
            target = ops.to_tensor(data_source.output)
        pipe.build([data_source.spec, target.spec], [target.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.randint(0, 255, (1, 3, 1080, 1920), dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as context:
            inputs = {data_source.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_to_tensor_with_different_layout_or_shape_tensor(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            data_source = ops.external_source("Source")
            target = ops.to_tensor(data_source.output)
        pipe.build([data_source.spec, target.spec], [target.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 1080, 1920, 3), dtype=torch.float)
        with self.assertRaises(RuntimeError) as context:
            inputs = {data_source.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_to_tensor_with_input_exceed_output(self):
        pipe = Pipeline(batch_size=1, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            data_source = ops.external_source("Source")
            target = ops.to_tensor(data_source.output)
        pipe.build([data_source.spec, target.spec], [target.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        with self.assertRaises(RuntimeError) as context:
            inputs = {data_source.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_to_tensor_with_spec_invalid_layout(self):
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        # Build pipeline
        with pipe:
            data_source = ops.external_source("Source")
            target = ops.to_tensor(data_source.output, TensorLayout.PLAIN)
        pipe.build([data_source.spec, target.spec], [target.output])

        # Run pipeline
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        with self.assertRaises(RuntimeError) as context:
            inputs = {data_source.output.name: to_accdata_tensorlist([tensor])}
            pipe.run(**inputs)
        self.assertIn("Pipeline run failed with error code", str(context.exception))


def basic_to_tensor_success(thread_cnt, input_data, target_layout):
    pipe = Pipeline(batch_size=32, num_threads=thread_cnt, queue_depth=5)

    with pipe:
        data_source = ops.external_source("Source")
        if target_layout:
            target = ops.to_tensor(data_source.output, layout=target_layout)
        else:
            # Use default
            target = ops.to_tensor(data_source.output)
            target_layout = _t.TensorLayout.NCHW
        pipe.build([data_source.spec, target.spec], [target.output])

    for source in input_data:
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        accdata_outputs = pipe.run(**feed_data)
        assert accdata_outputs
        accdata_torch_outputs = to_torch_tensor(accdata_outputs[0][0])
        expect = TorchOpTransforms.to_tensor_with_layout(source.tensor.clone(), source.layout, target_layout)
        assert TorchOpTransforms.compare_tensors(accdata_torch_outputs, expect)


@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("target_layout", [_t.TensorLayout.NCHW, _t.TensorLayout.NHWC, None])
@pytest.mark.parametrize("input_data", [RandomDataSource.data_uint8_nchw, RandomDataSource.data_uint8_nhwc])
@pytest.mark.smoke
@pytest.mark.to_tensor
def test_to_tensor_smoke(thread_cnt, input_data, target_layout):
    basic_to_tensor_success(thread_cnt, input_data, target_layout)


@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("target_layout", [_t.TensorLayout.NCHW, _t.TensorLayout.NHWC, None])
@pytest.mark.parametrize("input_data",
                        [RandomDataSource.data_uint8_nchw_random_shape, RandomDataSource.data_uint8_nhwc_random_shape])
@pytest.mark.slow
@pytest.mark.to_tensor
def test_to_tensor_precision(thread_cnt, input_data, target_layout):
    RandomDataSource.create_random_data_source_slow()
    basic_to_tensor_success(thread_cnt, input_data, target_layout)


if __name__ == '__main__':
    unittest.main()

