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


def to_tensor_torch(data_source, thread_num, dst_layout):
    return TorchOpTransforms.to_tensor_with_layout(data_source.tensor, data_source.layout,
                                                dst_layout)


def to_tensor_accdata(input_source, thread_num, dst_layout):
    pipe = Pipeline(num_threads=thread_num)
    with pipe:
        data_source = ops.external_source("Source")
        target = ops.to_tensor(data_source.output, dst_layout)
        pipe.build([data_source.spec, target.spec], [target.output])
    inputs = {data_source.output.name: to_accdata_tensorlist([input_source.tensor])}
    outputs = pipe.run(**inputs)
    return to_torch_tensor(outputs[0][0])


@pytest.mark.parametrize("to_tensor_mode", [to_tensor_torch, to_tensor_accdata])
@pytest.mark.parametrize("data_source", [RandomDataSource.data_uint8_nhwc[0], RandomDataSource.data_uint8_nchw[0]],
                         ids=["1080p_nhwc", "1080p_nchw"])
@pytest.mark.parametrize("dst_layout", [_t.TensorLayout.NHWC, _t.TensorLayout.NCHW], ids=["to_NHWC", "to_NCHW"])
@pytest.mark.parametrize("thread_num", [1])
def test_to_tensor(benchmark, to_tensor_mode, data_source, dst_layout, thread_num):
    TorchOpTransforms.prepare_torch_thread(thread_num)
    benchmark(to_tensor_mode, data_source, thread_num, dst_layout)