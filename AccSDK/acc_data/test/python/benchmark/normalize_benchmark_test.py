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
import pytest
import accdata.ops as ops
import accdata.types as _t
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensorlist
from ut.utils import RandomDataSource, TorchOpTransforms


class NormalizeArgs:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

normalize_args = NormalizeArgs(mean=0.485, std=0.229)


def normalize_torch(data_source, thread_num):
    return TorchOpTransforms.normalize(data_source.tensor, normalize_args)


def normalize_accdata(input_source, thread_num):
    pipe = Pipeline(num_threads=thread_num)
    with pipe:
        data_source = ops.external_source("Source")
        norm = ops.normalize(data_source.output, mean=[normalize_args.mean, normalize_args.mean, normalize_args.mean],
                                 std=[normalize_args.std, normalize_args.std, normalize_args.std])
        pipe.build([data_source.spec, norm.spec], [norm.output])
    inputs = {data_source.output.name: to_accdata_tensorlist([input_source.tensor])}
    outputs = pipe.run(**inputs)
    return to_torch_tensorlist(outputs[0])


@pytest.mark.parametrize("normalize_mode", [normalize_torch, normalize_accdata])
@pytest.mark.parametrize("data_source", [RandomDataSource.data_float_nhwc[0], RandomDataSource.data_float_nchw[0]],
                         ids=["1080p_nhwc", "1080p_nchw"])
@pytest.mark.parametrize("thread_num", [1])
def test_normalize(benchmark, normalize_mode, data_source, thread_num):
    TorchOpTransforms.prepare_torch_thread(thread_num)
    benchmark(normalize_mode, data_source, thread_num)