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
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensor
from ut.utils import RandomDataSource, TorchOpTransforms


class OpenSoraArgs:
    def __init__(self, mean, std, target_size):
        self.mean = mean
        self.std = std
        self.target_size = target_size
        self.interpolation_mode = "bilinear"

open_sora_args = OpenSoraArgs(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    target_size=(1024, 1024)
)


def opensora_torch(data_source, thread_num):
    return TorchOpTransforms.to_tensor_resize_crop_normalize(data_source.tensor, open_sora_args)


def opensora_accdata(input_source, thread_num):
    pipe = Pipeline(num_threads=thread_num)
    with pipe:
        data_source = ops.external_source("Source")
        fusion_ret = ops.to_tensor_resize_crop_norm(
            data_source.output,
            resize=open_sora_args.target_size,
            mean=open_sora_args.mean, std=open_sora_args.std
        )
    pipe.build([data_source.spec, fusion_ret.spec], [fusion_ret.output])
    inputs = {data_source.output.name: to_accdata_tensorlist([input_source.tensor])}
    outputs = pipe.run(**inputs)
    return to_torch_tensor(outputs[0][0])


@pytest.mark.parametrize("opensora_mode", [opensora_torch, opensora_accdata])
@pytest.mark.parametrize("data_source", [RandomDataSource.data_uint8_nhwc[0]], ids=["1080p_nhwc"])
@pytest.mark.parametrize("thread_num", [1, 8])
def test_opensora_op(benchmark, opensora_mode, data_source, thread_num):
    TorchOpTransforms.prepare_torch_thread(thread_num)
    benchmark(opensora_mode, data_source, thread_num)