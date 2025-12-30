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
import accdata.backend as _b
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import to_accdata_tensorlist, to_torch_tensorlist
from ut.utils import RandomDataSource, TorchOpTransforms


class ResizeCropArgs:
    def __init__(self, target_size, interpolation_mode):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode


def resize_crop_torch(data_source, thread_num, interpolation_mode):
    resize_crop_args = ResizeCropArgs(data_source.target_size, interpolation_mode)
    return TorchOpTransforms.resize_crop(data_source.tensor, resize_crop_args)


def resize_crop_accdata(input_source, thread_num, interpolation_mode):
    resize_size = input_source.get_resize_pos()
    target_size = input_source.target_size
    pipe = Pipeline(num_threads=thread_num)
    with pipe:
        image_data = ops.external_source("images")
        resize_crop = ops.resize_crop(image_data.output, resize_size, target_size,
                                        interpolation_mode=interpolation_mode)
    pipe.build([image_data.spec, resize_crop.spec], [resize_crop.output])
    input_data = {image_data.output.name: to_accdata_tensorlist([input_source.tensor])}
    outputs = pipe.run(**input_data)
    return to_torch_tensorlist(outputs[0])


@pytest.mark.parametrize("resize_crop_mode", [resize_crop_torch, resize_crop_accdata])
@pytest.mark.parametrize("data_source", [RandomDataSource.data_float_nchw[0]], ids=["1080p_nchw"])
@pytest.mark.parametrize("interpolation_mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("thread_num", [1, 8])
def test_resize_crop(benchmark, resize_crop_mode, thread_num, data_source, interpolation_mode):
    TorchOpTransforms.prepare_torch_thread(thread_num)
    benchmark(resize_crop_mode, data_source, thread_num, interpolation_mode)