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
import random
import pytest
import torch
from transformers.image_processing_utils import BatchFeature
from transformers import Qwen2VLImageProcessor

import accdata.backend as _b
import accdata.types as _t
from accdata import ops
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensorlist

from ut.utils import RandomDataSource, TorchOpTransforms


@pytest.mark.smoke
@pytest.mark.qwen
class QwenFusedOpTest(unittest.TestCase):
    def setUp(self):
        _b.SetLogLevel("error")
        self.image_mean = [random.uniform(0, 1) for _ in range(3)]
        self.image_std = [random.uniform(0, 1) for _ in range(3)]
        self.min_pixels = 3136
        self.max_pixels = 12845056
        self.patch_size = 14
        self.temporal_patch_size = 2
        self.merge_size = 2

    def accdata_qwen_fused_op(self, batch_size, thread_cnt, queue_depth, input_data):
        for source in input_data:
            if self.merge_size * self.patch_size > source.source_shape[0] or \
                self.merge_size * self.patch_size > source.source_shape[1] or \
                source.source_shape[0] != 1:
                continue
            pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.qwen_fusion_op(
                    data_source.output, mean=self.image_mean, std=self.image_std, min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels, patch_size=self.patch_size,
                    temporal_patch_size=self.temporal_patch_size, merge_size=self.merge_size)
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            outputs = pipe.run(**feed_data)
            accdata_ret = to_torch_tensorlist(outputs[0])
            accdata_data = {}
            accdata_data['pixel_values'] = accdata_ret[0]
            accdata_data['image_grid_thw'] = torch.tensor([[1, 78, 138]], dtype=torch.int64)
            accdata_data = BatchFeature(data=accdata_data, tensor_type="pt")

            origin_ret = self.origin_transfomers_processor(source.tensor.clone())
            self.assertTrue(TorchOpTransforms.compare_tensors(
                accdata_data['pixel_values'], origin_ret))

    def origin_transfomers_processor(self, image_torch, patch_size=None, merge_size=None):
        parms = {
            "do_resize": True,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "patch_size": self.patch_size if patch_size is None else patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size if merge_size is None else merge_size,
        }
        img_ori_proc = Qwen2VLImageProcessor(**parms)

        image = image_torch.squeeze().numpy()
        ori_ret = img_ori_proc([image], return_tensors="pt")
        return ori_ret['pixel_values']

    def prepare_pipeline(self, mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711), min_pixels=3136, max_pixels=12845056,
        patch_size=14, temporal_patch_size=2, merge_size=2):
        pipe = Pipeline(batch_size=8, num_threads=4, queue_depth=2)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.qwen_fusion_op(
                data_source.output, mean=mean, std=std, min_pixels=min_pixels,
                max_pixels=max_pixels, patch_size=patch_size,
                temporal_patch_size=temporal_patch_size, merge_size=merge_size)
        pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])

        return pipe, data_source

    def test_qwen_default_input(self):
        for source in RandomDataSource.data_uint8_nhwc:
            if source.source_shape[0] != 1:
                continue
            pipe = Pipeline(batch_size=8, num_threads=4, queue_depth=2)
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.qwen_fusion_op(
                    data_source.output, mean=self.image_mean, std=self.image_std, max_pixels=self.max_pixels)
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            outputs = pipe.run(**feed_data)
            accdata_ret = to_torch_tensorlist(outputs[0])
            accdata_data = {}
            accdata_data['pixel_values'] = accdata_ret[0]
            accdata_data['image_grid_thw'] = torch.tensor([[1, 78, 138]], dtype=torch.int64)
            accdata_data = BatchFeature(data=accdata_data, tensor_type="pt")

            origin_ret = self.origin_transfomers_processor(source.tensor.clone())
            self.assertTrue(TorchOpTransforms.compare_tensors(
                accdata_data['pixel_values'], origin_ret))
    
    def test_qwen_input_type_error(self):
        source = RandomDataSource.data_float_nhwc[0]
        pipe, data_source = self.prepare_pipeline()
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_qwen_input_layout_error(self):
        source = RandomDataSource.data_uint8_nchw[0]
        pipe, data_source = self.prepare_pipeline()
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_qwen_input_empty(self):
        pipe, _ = self.prepare_pipeline()
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run()
        self.assertIn("Pipeline run failed", str(context.exception))
            
    def test_mean_std_length_error(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        pipe, data_source = self.prepare_pipeline(mean=[0.5, 0.5], std=[0.5])
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_pixel_value_error(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        pipe, data_source = self.prepare_pipeline(min_pixels=12845056, max_pixels=3136)
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_pixel_value_correct(self):
        pipe, data_source = self.prepare_pipeline(min_pixels=10*10, max_pixels=8192*8192)
        assert pipe is not None
        assert data_source is not None

    def test_patch_size_error(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        pipe, data_source = self.prepare_pipeline(patch_size=-1)
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_merge_size_error(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        pipe, data_source = self.prepare_pipeline(merge_size=-1)
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_pixel_value_extremum(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        pipe, data_source = self.prepare_pipeline(min_pixels=10*10 - 1, max_pixels=8192*8192 + 1)
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_patch_size_greater_than_input(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        pipe, data_source = self.prepare_pipeline(patch_size=2048)
        feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
        with self.assertRaises(RuntimeError) as context:
            _ = pipe.run(**feed_data)
        self.assertIn("Pipeline run failed", str(context.exception))

    def test_big_patch_size_success(self):
        for source in RandomDataSource.data_uint8_nhwc:
            if source.source_shape[0] != 1:
                continue
            pipe = Pipeline(batch_size=8, num_threads=4, queue_depth=2)
            input_patch_size = 720
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.qwen_fusion_op(
                    data_source.output, mean=self.image_mean, std=self.image_std, max_pixels=self.max_pixels,
                    patch_size=input_patch_size, merge_size=1)
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            outputs = pipe.run(**feed_data)
            accdata_ret = to_torch_tensorlist(outputs[0])
            accdata_data = {}
            accdata_data['pixel_values'] = accdata_ret[0]
            accdata_data['image_grid_thw'] = torch.tensor([[1, 78, 138]], dtype=torch.int64)
            accdata_data = BatchFeature(data=accdata_data, tensor_type="pt")

            origin_ret = self.origin_transfomers_processor(source.tensor.clone(),
                patch_size=input_patch_size, merge_size=1)
            self.assertTrue(TorchOpTransforms.compare_tensors(
                accdata_data['pixel_values'], origin_ret))

    def test_qwen_fuse_op_with_different_type_tensor(self):
        pipe = Pipeline(batch_size=4, num_threads=4, queue_depth=2)
        input_patch_size = 720
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.randint(0, 255, (1, 3, 1080, 1920), dtype=torch.uint8)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.qwen_fusion_op(
                data_source.output, mean=self.image_mean, std=self.image_std, max_pixels=self.max_pixels,
                patch_size=input_patch_size, merge_size=1)
        pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
        with self.assertRaises(RuntimeError) as context:
            feed_data = {data_source.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**feed_data)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_qwen_fuse_op_with_different_layout_or_shape_tensor(self):
        pipe = Pipeline(batch_size=4, num_threads=4, queue_depth=2)
        input_patch_size = 720
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 1080, 1920, 3), dtype=torch.float)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.qwen_fusion_op(
                data_source.output, mean=self.image_mean, std=self.image_std, max_pixels=self.max_pixels,
                patch_size=input_patch_size, merge_size=1)
        pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
        with self.assertRaises(RuntimeError) as context:
            feed_data = {data_source.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**feed_data)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_qwen_fuse_op_with_input_exceed_output(self):
        pipe = Pipeline(batch_size=1, num_threads=4, queue_depth=2)
        input_patch_size = 720
        tensor = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        tensor1 = torch.rand((1, 3, 1080, 1920), dtype=torch.float)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.qwen_fusion_op(
                data_source.output, mean=self.image_mean, std=self.image_std, max_pixels=self.max_pixels,
                patch_size=input_patch_size, merge_size=1)
        pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
        with self.assertRaises(RuntimeError) as context:
            feed_data = {data_source.output.name: to_accdata_tensorlist([tensor, tensor1])}
            pipe.run(**feed_data)
        self.assertIn("Pipeline run failed with error code", str(context.exception))


fused_op_suits = QwenFusedOpTest()
fused_op_suits.setUp()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("queue_depth", [4])
@pytest.mark.smoke
@pytest.mark.qwen
def test_fused_op_success(batch_size, thread_cnt, queue_depth):
    fused_op_suits.accdata_qwen_fused_op(batch_size, thread_cnt, queue_depth, RandomDataSource.data_uint8_nhwc)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("queue_depth", [4])
@pytest.mark.slow
def test_fused_op_precision(batch_size, thread_cnt, queue_depth):
    RandomDataSource.create_random_data_source_slow()
    fused_op_suits.accdata_qwen_fused_op(batch_size, thread_cnt, queue_depth,
                                         RandomDataSource.data_uint8_nhwc_random_shape)


if __name__ == '__main__':
    unittest.main()
