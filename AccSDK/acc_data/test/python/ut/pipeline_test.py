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
import copy

import pytest

import accdata.backend as _b
from accdata import ops
from accdata.pipeline import Pipeline, pipeline_def
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensor

from ut.utils import RandomDataSource, TorchOpTransforms


@pytest.mark.smoke
class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.datas_int = RandomDataSource.data_uint8_nhwc[0]
        self.datas_float = RandomDataSource.data_float_nchw[0]
        # resize_crop
        self.origin_size = (1080, 1920)
        self.interpolation_mode = "bilinear"
        self.target_size = (1024, 1024)
        rh, rw = self.target_size[0] / \
            self.origin_size[0], self.target_size[1] / self.origin_size[1]
        if rh > rw:
            self.resize_pos = self.target_size[0], round(
                self.origin_size[1] * rh)
        else:
            self.resize_pos = round(
                self.origin_size[0] * rw), self.target_size[1]

        # normalize
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def do_test_pipe(self, pipe, external_data, data, torch_op):
        input_data = {
            external_data.name:
            to_accdata_tensorlist([copy.deepcopy(data.tensor)])
        }
        accdata_outputs = pipe.run(**input_data)
        self.assertIsNotNone(accdata_outputs)

        accdata_torch_outputs = to_torch_tensor(accdata_outputs[0][0])
        torch_origin_outputs = torch_op(data.tensor, self)
        self.assertTrue(
            TorchOpTransforms.compare_tensors(accdata_torch_outputs,
                                                torch_origin_outputs))

    def accdata_to_tensor(self, pipe):
        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            pipe.build([external_data.spec, to_tensor_frames.spec], [to_tensor_frames.output])
        self.do_test_pipe(pipe, external_data.output, self.datas_int,
                          TorchOpTransforms.to_tensor)

    def accdata_normal(self, pipe):
        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            normalize_frames = ops.normalize(external_data.output,
                                             mean=self.mean,
                                             std=self.std)
            pipe.build([external_data.spec, normalize_frames.spec], [normalize_frames.output])
        self.do_test_pipe(pipe, external_data.output, self.datas_float,
                          TorchOpTransforms.normalize)

    def accdata_resize_crop(self, pipe):
        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            resize_crop_frames = ops.resize_crop(external_data.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            pipe.build([external_data.spec, resize_crop_frames.spec], [resize_crop_frames.output])

        self.do_test_pipe(pipe, external_data.output, self.datas_float,
                          TorchOpTransforms.resize_crop)

    def accdata_to_tensor_resize_crop_normalize(self, pipe):
        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)
            pipe.build([external_data.spec, to_tensor_frames.spec, resize_crop_frames.spec, normalize_frames.spec],
                       [normalize_frames.output])

        self.do_test_pipe(pipe, external_data.output, self.datas_int,
                          TorchOpTransforms.to_tensor_resize_crop_normalize)

    def test_operation_build_with_none(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)
        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)

        with self.assertRaises(TypeError) as context:
            pipe.build(None, None)
            self.assertIn("Unsupported input type", str(context.exception))

    def test_operation_build_with_empty_specs(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)

        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)

        with self.assertRaises(RuntimeError) as context:
            pipe.build([], [normalize_frames.output])
            self.assertIn("No datanode found for output.", str(context.exception))

    def test_operation_build_with_empty_outputs(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)

        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)

        with self.assertRaises(RuntimeError) as context:
            pipe.build([external_data.spec, to_tensor_frames.spec, resize_crop_frames.spec, normalize_frames.spec], [])
            self.assertIn("No path to generate the outputs", str(context.exception))

    def test_to_tensor_normal_cut(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=False)

        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            _ = ops.normalize(resize_crop_frames.output,
                              mean=self.mean,
                              std=self.std)
            pipe.build([external_data.spec, to_tensor_frames.spec, resize_crop_frames.spec, _.spec],
                       [resize_crop_frames.output])

        self.do_test_pipe(pipe, external_data.output, self.datas_int,
                          TorchOpTransforms.to_tensor_resize_crop)

    def test_operation_not_used(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=False)

        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            to_tensor_frames_2 = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            resize_crop_frames_2 = ops.resize_crop(to_tensor_frames_2.output,
                                                   resize=self.resize_pos,
                                                   crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)
            normalize_frames_2 = ops.normalize(resize_crop_frames_2.output, mean=self.mean, std=self.std)
            pipe.build([external_data.spec, to_tensor_frames.spec, to_tensor_frames_2.spec, resize_crop_frames.spec,
                        resize_crop_frames_2.spec, normalize_frames.spec, normalize_frames_2.spec],
                       [normalize_frames.output])

        self.do_test_pipe(pipe, external_data.output, self.datas_int,
                          TorchOpTransforms.to_tensor_resize_crop_normalize)

    def test_pipeline_auto_fusion(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)

        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)
            pipe.build([external_data.spec, to_tensor_frames.spec, resize_crop_frames.spec, normalize_frames.spec],
                       [normalize_frames.output])

        self.do_test_pipe(pipe, external_data.output, self.datas_int,
                          TorchOpTransforms.to_tensor_resize_crop_normalize)

    def test_pipeline_run_error_input(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)

        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            normalize_frames = ops.normalize(external_data.output,
                                             mean=self.mean,
                                             std=self.std)
            pipe.build([external_data.spec, normalize_frames.spec], [normalize_frames.output])

        test_intput_values = [
            ["hello"],
            [123],
            [123.],
            self.datas_float,
            self.datas_int,
            123,
            123.
        ]

        for data in test_intput_values:
            input_data = {external_data.output.name: data}
            try:
                _ = pipe.run(**input_data)
            except Exception as e:
                self.assertEqual(type(e), TypeError)
                self.assertIn("Unsupported input type", str(e))

    def test_pipeline_build_error_input(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)

        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            normalize_frames = ops.normalize(external_data.output,
                                             mean=self.mean,
                                             std=self.std)
        test_build_type_error = [
            123,
            123.,
            {"tom": 1},
            True,
            None
        ]

        test_build_runtime_error = [
            [1231, True],
            [[]],
            [{}]
        ]
        for data in test_build_type_error:
            try:
                _ = pipe.build([], data)
            except Exception as e:
                self.assertEqual(type(e), TypeError)
                self.assertIn("Unsupported input type", str(e))

        for data in test_build_runtime_error:
            try:
                _ = pipe.build([], data)
            except Exception as e:
                self.assertEqual(type(e), RuntimeError)
                self.assertIn("Expected a string or an object", str(e))
    
    def test_pipeline_error_twice(self):
        pipe = Pipeline(batch_size=32,
                        num_threads=8,
                        queue_depth=10,
                        auto_fuse=True)

        external_data = None
        with pipe:
            external_data = ops.external_source("external_data")
            normalize_frames = ops.normalize(external_data.output,
                                             mean=self.mean,
                                             std=self.std)
            pipe.build([external_data.spec, normalize_frames.spec], [normalize_frames.output])
        external_error = ops.external_source("external_error")

        input_data = {
            external_error.output.name:
            to_accdata_tensorlist([copy.deepcopy(self.datas_float.tensor)])
        }
        try:
            _ = pipe.run(**input_data)
        except Exception as e:
            self.assertEqual(type(e), RuntimeError)
            self.assertIn(str(_b.ErrorCode.H_SINGLEOP_ERROR.value), str(e)) # norm operation error
        
        try:
            _ = pipe.run(**input_data)
        except Exception as e:
            self.assertEqual(type(e), RuntimeError)
            self.assertIn(str(_b.ErrorCode.H_PIPELINE_STATE_ERROR.value), str(e)) # norm operation error

    def test_pipeline_def_success(self):
        @pipeline_def
        def to_tensor_norm_resize_crop_pipe():
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)
            op_specs = [external_data.spec, to_tensor_frames.spec, resize_crop_frames.spec, normalize_frames.spec]
            pipe_outputs = [normalize_frames.output]
            input_nodes = [external_data]
            return input_nodes, op_specs, pipe_outputs
        pipe, input_nodes = to_tensor_norm_resize_crop_pipe(batch_size=2, num_threads=3, queue_depth=4)
        inputs = {
            input_nodes[0].output.name:
            to_accdata_tensorlist([copy.deepcopy(self.datas_int.tensor)])
        }
        outputs = pipe.run(**inputs)
        self.assertIsNotNone(outputs)

    def test_pipeline_def_error_output(self):
        @pipeline_def
        def to_tensor_norm_resize_crop_pipe():
            external_data = ops.external_source("external_data")
            to_tensor_frames = ops.to_tensor(external_data.output)
            resize_crop_frames = ops.resize_crop(to_tensor_frames.output,
                                                 resize=self.resize_pos,
                                                 crop=self.target_size)
            normalize_frames = ops.normalize(resize_crop_frames.output,
                                             mean=self.mean,
                                             std=self.std)
            op_specs = [external_data.spec, to_tensor_frames.spec, resize_crop_frames.spec, normalize_frames.spec]
            pipe_outputs = [normalize_frames.output]
            return op_specs, pipe_outputs
        with self.assertRaises(RuntimeError) as context:
            _ = to_tensor_norm_resize_crop_pipe(batch_size=2, num_threads=3, queue_depth=4)
        self.assertIn("Pipeline outputs should be a tuple of tree element", str(context.exception))

pipeline_suits = PipelineTest()
pipeline_suits.setUp()


@pytest.mark.parametrize("batch_size", [1, 8, 256])
@pytest.mark.parametrize("thread_cnt", [1, 2, 3])
@pytest.mark.parametrize("queue_depth", [4, 32, 64])
@pytest.mark.parametrize("fuse_choice", [True, False])
@pytest.mark.smoke
def test_pipeline_success(batch_size, thread_cnt, queue_depth, fuse_choice):
    pipe = Pipeline(batch_size=batch_size,
                num_threads=thread_cnt,
                queue_depth=queue_depth,
                auto_fuse=fuse_choice)
    pipeline_suits.accdata_to_tensor_resize_crop_normalize(pipe)


@pytest.mark.parametrize("batch_size", [-1, 0, 1025])
@pytest.mark.parametrize("thread_cnt", [-1, 0, 1025])
@pytest.mark.parametrize("queue_depth", [-1, 1, 129])
@pytest.mark.smoke
def test_pipeline_input_default(batch_size, thread_cnt, queue_depth):
    pipe = Pipeline(batch_size=batch_size,
                num_threads=thread_cnt,
                queue_depth=queue_depth,
                auto_fuse=True)
    assert pipe._pipe is None

if __name__ == "__main__":
    unittest.main()
