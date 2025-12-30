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
from accdata import ops
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensor

from ut.utils import RandomDataSource, TorchOpTransforms, HandleSource


@pytest.mark.smoke
class FusedOpTest(unittest.TestCase):
    def setUp(self):
        _b.SetLogLevel("error")
        # resize_crop
        self.interpolation_mode = "bilinear"

        # normalize
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def accdata_fused_op_with_default_args(self, batch_size, thread_cnt, queue_depth, input_data):
        for source in input_data:
            resize_pos = source.get_resize_pos()
            crop_pos = source.target_size
            if not TorchOpTransforms.check_image_size(resize_pos) or\
                not TorchOpTransforms.check_image_size(crop_pos):
                continue
            pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std
                )
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            self.run_pipeline_with_accuracy_check(feed_data, pipe, source, crop_size=crop_pos, resize_size=resize_pos)

    def accdata_fused_op_with_boundary_args(self, resize_size, crop_size, input_data):
        for source in input_data:
            pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_size,
                    crop=crop_size, round_mode="truncate",
                    mean=self.mean, std=self.std
                )
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            self.run_pipeline_with_accuracy_check(feed_data, pipe, source, crop_size=crop_size, resize_size=resize_size)

    def accdata_fused_op(self, batch_size, thread_cnt, queue_depth, input_data):
        for source in input_data:
            resize_pos = source.get_resize_pos()
            crop_pos = source.target_size
            if not TorchOpTransforms.check_image_size(resize_pos) or\
                not TorchOpTransforms.check_image_size(crop_pos):
                continue
            pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode=self.interpolation_mode,
                    crop_pos_w=0.5, crop_pos_h=0.5, layout=_t.TensorLayout.NCHW
                )
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            self.run_pipeline_with_accuracy_check(feed_data, pipe, source, crop_size=crop_pos, resize_size=resize_pos)

    def accdata_fused_op_with_none_crop(self, batch_size, thread_cnt, queue_depth, input_data):
        for source in input_data:
            resize_pos = source.get_resize_pos()
            crop_pos = None
            pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
            with pipe:
                data_source = ops.external_source("Source")
                fusion_ret = ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode=self.interpolation_mode,
                    crop_pos_w=0.5, crop_pos_h=0.5, layout=_t.TensorLayout.NCHW
                )
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            self.run_pipeline_with_accuracy_check_with_none_crop(feed_data, pipe, source)

    def run_pipeline_with_accuracy_check(self, feed_data, pipe, source, crop_size, resize_size):
        accdata_outputs = pipe.run(**feed_data)
        self.assertIsNotNone(accdata_outputs)
        self.assertIsNotNone(accdata_outputs[0])
        accdata_torch_outputs = to_torch_tensor(accdata_outputs[0][0])
        self.target_size = crop_size
        expect = TorchOpTransforms.to_tensor_resize_crop_normalize(source.tensor.clone(), self, resize_size)
        self.assertTrue(
            TorchOpTransforms.compare_tensors(accdata_torch_outputs,
                                              expect))

    def run_pipeline_with_accuracy_check_with_none_crop(self, feed_data, pipe, source):
        accdata_outputs = pipe.run(**feed_data)
        self.assertIsNotNone(accdata_outputs)
        self.assertIsNotNone(accdata_outputs[0])
        accdata_torch_outputs = to_torch_tensor(accdata_outputs[0][0])
        self.target_size = source.get_resize_pos()
        expect = TorchOpTransforms.to_tensor_resize_crop_normalize(source.tensor.clone(), self)
        self.assertTrue(
            TorchOpTransforms.compare_tensors(accdata_torch_outputs,
                                              expect))

    def test_fused_op_with_invalid_tensor_type(self):
        source = RandomDataSource.data_float_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_unsupported_interpolation_mode(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bicubic"
                )
            self.assertIn("Expected 'bilinear', received 'bicubic'", str(context.exception))

    def test_fused_op_with_invalid_interpolation_mode(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="invalid_type"
                )
            self.assertIn("Expected 'bilinear', received 'invalid_type'", str(context.exception))

    def test_fused_op_with_unsupported_round_mode(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="invalid_type",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Expected '('round', 'truncate')', received 'invalid_type'", str(context.exception))

    def test_fused_op_with_invalid_input(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            _input = {"bad_key": 1}
            with self.assertRaises(TypeError) as context:
                ops.to_tensor_resize_crop_norm(
                    _input,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn(f"Expected inputs of type 'DataNode'. Received input of type '{type(_input)}'.",
                          str(context.exception))

    def test_fused_op_with_resize_none(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = None
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_fused_op_with_crop_height_greater_than_resize_height(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = [resize_pos[0] + 1, resize_pos[1]]
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("crop height size cannot greater than resize height size", str(context.exception))

    def test_fused_op_with_crop_width_greater_than_resize_width(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = [resize_pos[0], resize_pos[1] + 1]
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("crop width size cannot greater than resize width size", str(context.exception))

    def test_fused_op_with_invalid_resize_type(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = {"bad_key": 1}
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_fused_op_with_invalid_resize_elements(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1024, "1024")
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_fused_op_with_invalid_resize_size(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1024, 1024, 1024)
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid resize input!", str(context.exception))

    def test_fused_op_with_invalid_crop_type(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = {"bad_key": 1}
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid crop input!", str(context.exception))

    def test_fused_op_with_invalid_crop_elements(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = [1024, "1024"]
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid crop input!", str(context.exception))

    def test_fused_op_with_invalid_crop_size(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = [1024, 1024, 1024]
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            with self.assertRaises(ValueError) as context:
                ops.to_tensor_resize_crop_norm(
                    data_source.output,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn("Invalid crop input!", str(context.exception))

    def test_fused_op_with_resize_height_less_than_10(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (5, 1024)
        crop_pos = (5, 1024)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_resize_height_larger_than_8192(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (8193, 1024)
        crop_pos = (8193, 1024)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_resize_width_less_than_10(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1024, 5)
        crop_pos = (1024, 5)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_resize_width_larger_than_8192(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1024, 8193)
        crop_pos = (1024, 8193)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_crop_height_less_than_zero(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1080, 1980)
        crop_pos = (-1, 1024)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_crop_width_less_than_zero(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1080, 1980)
        crop_pos = (1024, -1)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_crop_height_equals_to_zero(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1080, 1980)
        crop_pos = (0, 1024)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_crop_width_equals_to_zero(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = (1080, 1980)
        crop_pos = (1024, 0)
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_output_layout_nhwc(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std, interpolation_mode=self.interpolation_mode,
                crop_pos_w=0.5, crop_pos_h=0.5, layout=_t.TensorLayout.NHWC
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))
    
    def test_fused_op_with_input_layout_nchw(self):
        source = RandomDataSource.data_uint8_nchw[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std, interpolation_mode=self.interpolation_mode,
                crop_pos_w=0.5, crop_pos_h=0.5, layout=_t.TensorLayout.NCHW
            )
        with self.assertRaises(RuntimeError) as context:
            pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
            feed_data = {data_source.output.name: to_accdata_tensorlist([source.tensor.clone()])}
            pipe.run(**feed_data)
            self.assertIn("Pipeline run failed", str(context.exception))

    def test_fused_op_with_none_input(self):
        source = RandomDataSource.data_uint8_nhwc[0]
        resize_pos = source.get_resize_pos()
        crop_pos = source.target_size
        pipe = Pipeline(batch_size=32, num_threads=8, queue_depth=10)
        with pipe:
            _input = None
            with self.assertRaises(TypeError) as context:
                ops.to_tensor_resize_crop_norm(
                    _input,
                    resize=resize_pos,
                    crop=crop_pos, round_mode="truncate",
                    mean=self.mean, std=self.std, interpolation_mode="bilinear"
                )
            self.assertIn(f"Expected inputs of type 'DataNode'. Received input of type '{type(_input)}'.",
                          str(context.exception))

    def test_fused_op_with_different_type_tensor(self):
        batch_size = 8
        thread_cnt = 8
        queue_depth = 4
        source = HandleSource((1, 1080, 1920, 3), (1024, 1024), torch.uint8, _t.TensorLayout.NHWC, "1080p")
        source1 = HandleSource((1, 1080, 1920, 3), (1024, 1024), torch.float, _t.TensorLayout.NHWC, "1080p")
        resize_pos = source.get_resize_pos()
        crop_pos = None
        pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std, interpolation_mode=self.interpolation_mode,
                crop_pos_w=0.5, crop_pos_h=0.5, layout=_t.TensorLayout.NCHW
            )
        pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
        with self.assertRaises(RuntimeError) as context:
            feed_data = \
                {data_source.output.name: to_accdata_tensorlist([source.tensor.clone(), source1.tensor.clone()])}
            pipe.run(**feed_data)
        self.assertIn("Pipeline run failed with error code", str(context.exception))

    def test_fused_op_with_different_layout_or_shape_tensor(self):
        batch_size = 8
        thread_cnt = 8
        queue_depth = 4
        source = HandleSource((1, 1080, 1920, 3), (1024, 1024), torch.uint8, _t.TensorLayout.NHWC, "1080p")
        source1 = HandleSource((1, 3, 1080, 1920), (1024, 1024), torch.uint8, _t.TensorLayout.NHWC, "1080p")
        resize_pos = source.get_resize_pos()
        crop_pos = None
        pipe = Pipeline(batch_size=batch_size, num_threads=thread_cnt, queue_depth=queue_depth)
        with pipe:
            data_source = ops.external_source("Source")
            fusion_ret = ops.to_tensor_resize_crop_norm(
                data_source.output,
                resize=resize_pos,
                crop=crop_pos, round_mode="truncate",
                mean=self.mean, std=self.std, interpolation_mode=self.interpolation_mode,
                crop_pos_w=0.5, crop_pos_h=0.5, layout=_t.TensorLayout.NCHW
            )
        pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
        with self.assertRaises(RuntimeError) as context:
            feed_data = \
                {data_source.output.name: to_accdata_tensorlist([source.tensor.clone(), source1.tensor.clone()])}
            pipe.run(**feed_data)
        self.assertIn("Pipeline run failed with error code", str(context.exception))


fused_op_suits = FusedOpTest()
fused_op_suits.setUp()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("queue_depth", [4])
@pytest.mark.smoke
def test_fused_op_success(batch_size, thread_cnt, queue_depth):
    fused_op_suits.accdata_fused_op(batch_size, thread_cnt,
                                    queue_depth, RandomDataSource.data_uint8_nhwc)
    fused_op_suits.accdata_fused_op_with_default_args(batch_size, thread_cnt,
                                                      queue_depth, RandomDataSource.data_uint8_nhwc)
    fused_op_suits.accdata_fused_op_with_none_crop(batch_size, thread_cnt, queue_depth,
                                                   RandomDataSource.data_uint8_nhwc)


@pytest.mark.parametrize("resize_height", [20, 8192])
@pytest.mark.parametrize("resize_width", [20, 8192])
@pytest.mark.parametrize("crop_size_less", [0, 10])
@pytest.mark.smoke
def test_fused_op_success_with_boundary_params(resize_height, resize_width, crop_size_less):
    resize_size = [resize_height, resize_width]
    crop_size = [resize_height - crop_size_less, resize_width - crop_size_less]
    fused_op_suits.accdata_fused_op_with_boundary_args(resize_size, crop_size, RandomDataSource.data_uint8_nhwc)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("thread_cnt", [1, 8])
@pytest.mark.parametrize("queue_depth", [4])
@pytest.mark.slow
def test_fused_op_precision(batch_size, thread_cnt, queue_depth):
    RandomDataSource.create_random_data_source_slow()
    fused_op_suits.accdata_fused_op(batch_size, thread_cnt,
                                    queue_depth, RandomDataSource.data_uint8_nhwc_random_shape)
    fused_op_suits.accdata_fused_op_with_default_args(batch_size, thread_cnt,
                                                      queue_depth, RandomDataSource.data_uint8_nhwc_random_shape)


@pytest.mark.parametrize("resize_height", [20, 8192])
@pytest.mark.parametrize("resize_width", [20, 8192])
@pytest.mark.parametrize("crop_size_less", [0, 10])
@pytest.mark.slow
def test_fused_op_success_with_boundary_params_precision(resize_height, resize_width, crop_size_less):
    resize_size = [resize_height, resize_width]
    crop_size = [resize_height - crop_size_less, resize_width - crop_size_less]
    RandomDataSource.create_random_data_source_slow()
    fused_op_suits.accdata_fused_op_with_boundary_args(resize_size, crop_size,
                                                       RandomDataSource.data_uint8_nhwc_random_shape)


if __name__ == '__main__':
    unittest.main()
