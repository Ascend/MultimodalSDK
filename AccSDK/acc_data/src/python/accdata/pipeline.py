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

import logging

from functools import wraps
from inspect import signature

import accdata.backend as _backend
from accdata.data_node import DataNode as _DataNode


class Pipeline(object):
    """
    Pipeline is used to represent the data processing flow in AccData, whose duty is to expose the underlying
    C++ implement that responses for build the data processing graph and run the pipeline.

    Args:
        batch_size (int, optional, default = 1): how many samples per batch to load.
        queue_depth (int, optional, default = 1): prefetch queue length for the pipeline.
        num_threads (int, optional, default = 1): number of threads used in each data operation.
        auto_fuse (bool, optional, default = True): whether automatic replace some operations in fixed order to
            corresponding fusion operation or not.
    """
    _current = None

    def __init__(self,
                 batch_size=1,
                 queue_depth=2,
                 num_threads=1,
                 auto_fuse=True,
                 ):
        # parameter init
        self._batch_size = batch_size
        self._queue_depth = queue_depth
        self._num_threads = num_threads
        self._auto_fuse = auto_fuse
        self._logical_id = 0

        self._pipe = None
        self._set_backend_pipeline()

        self._output_data_nodes = None

    def __enter__(self):
        Pipeline._current = self
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def queue_depth(self):
        return self._queue_depth

    @property
    def num_threads(self):
        return self._num_threads

    @staticmethod
    def get_current():
        return Pipeline._current

    def create_datanode(self, name, device="cpu"):
        self._logical_id = self._logical_id + 1
        return _DataNode(name + "_" + str(self._logical_id), device, self)

    def build(self, specs, output_data_nodes=None):
        pipe_outputs = self._output_data_nodes if self._output_data_nodes is not None else output_data_nodes
        self._check_input(specs, list)
        for spec in specs:
            self._check_input(spec, _backend.OpSpec)
        self._check_input(pipe_outputs, list)
        err_code = self._pipe.Build(specs, pipe_outputs)
        self._check_ret(err_code)

    def run(self, **pipeline_inputs):
        self._check_input(pipeline_inputs, dict)
        for inp_name, inp_data in pipeline_inputs.items():
            self._check_input(inp_name, str)
            self._check_input(inp_data, _backend.TensorList)
        return self._pipe.Run(pipeline_inputs, False)

    def set_outputs(self, *outputs):
        self._output_data_nodes = outputs

    def _check_input(self, data, target):
        """
        :param data: User input data
        :param target: Expect user input target
        """
        if isinstance(data, target):
            return data

        raise TypeError(f"Unsupported input type {type(data)}")

    def _set_backend_pipeline(self):
        self._pipe = _backend.new_instance(self._batch_size, self._num_threads, self._queue_depth, self._auto_fuse)

    def _check_ret(self, err_code):
        if err_code != _backend.ErrorCode.H_OK:
            raise RuntimeError("Accdata backend runtime error: " + str(err_code) + "!")


def _distinguish_args(func, **func_kwargs):
    func_parms = signature(func).parameters
    init_parms = signature(Pipeline.__init__).parameters

    pipe_args = {}
    func_args = {}

    for arg_name, arg_value in func_kwargs.items():
        is_pipe_arg = arg_name in init_parms
        is_fn_arg = arg_name in func_parms
        if is_fn_arg:
            func_args[arg_name] = arg_value
            if is_pipe_arg:
                logging.info(
                    f"Warning: the argument {arg_name} shadows an argument of pipeline init with the same name.")
        elif is_pipe_arg:
            pipe_args[arg_name] = arg_value
        else:
            raise AssertionError(f"The argument '{arg_name}' is not supported.")

    return pipe_args, func_args


def _classify_args(func, pipeline_def_kwargs, fn_call_kwargs):
    """Classify arguments and identify which args are used for pipeline construction (pipeline kwargs)
    and which are used for pipeline definition function (function kwargs)"""
    pipe_args, func_kwargs = _distinguish_args(func, **fn_call_kwargs)
    updated_pipe_kwargs = {**pipeline_def_kwargs, **pipe_args}
    return updated_pipe_kwargs, func_kwargs


def pipeline_def(
        fn=None, **pipeline_kwargs
):
    """
    Decorator that converts a data process definition into an AccData pipeline.
    """
    @wraps(fn)
    def create_pipeline(*args, **kwargs):
        pipe_args, fn_kwargs = _classify_args(fn, pipeline_kwargs, kwargs)
        pipe = Pipeline(**pipe_args)
        with pipe:
            pipe_outputs = fn(*args, **fn_kwargs)
            if isinstance(pipe_outputs, tuple):
                try:
                    input_nodes, input_op_specs, pipe_outputs = pipe_outputs
                    pipe.build(input_op_specs, pipe_outputs)
                except Exception as e:
                    raise RuntimeError("Pipeline outputs should be a tuple of tree elements:"
                                       " (input_nodes, input_op_specs, pipe_outputs).") from e
        return pipe, input_nodes

    return create_pipeline
