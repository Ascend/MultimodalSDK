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
import torch
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_processing_utils import BatchFeature

import accdata.backend as _b
import accdata.types as _t
import accdata.ops as ops
from accdata.pipeline import Pipeline
from accdata.plugin.pytorch import (
    to_torch_tensorlist,
    to_accdata_tensorlist,
)
from ut.utils import TorchOpTransforms

IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
MIN_PIXELS = 3136
MAX_PIXELS = 518400
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
SIZE_FACTOR = 28


def accdata_processor(img, thread_num):
    pipe = Pipeline(num_threads=thread_num)
    with pipe:
        images_data = ops.external_source("images")
        fusion_ret = ops.qwen_fusion_op(
            images_data.output, mean=IMAGE_MEAN, std=IMAGE_STD, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
            patch_size=PATCH_SIZE, temporal_patch_size=TEMPORAL_PATCH_SIZE, merge_size=MERGE_SIZE)
        pipe.build([images_data.spec, fusion_ret.spec], [fusion_ret.output])
    ext_data = to_accdata_tensorlist([img.unsqueeze(0)], layout=_t.TensorLayout.NHWC)
    data = {images_data.output.name: ext_data}

    outputs = pipe.run(**data)
    accdata_ret = to_torch_tensorlist(outputs[0])

    accdata_data = {}
    accdata_data['pixel_values'] = accdata_ret[0]
    accdata_data['image_grid_thw'] = torch.tensor([[1, 78, 138]], dtype=torch.int64)
    accdata_data = BatchFeature(data=accdata_data, tensor_type="pt")
    return accdata_data


def transformers_processor(image, thread_num):
    parms = {
        "do_resize": True,
        "image_mean": IMAGE_MEAN,
        "image_std": IMAGE_STD,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "patch_size": PATCH_SIZE,
        "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        "merge_size": MERGE_SIZE,
    }
    img_ori_proc = Qwen2VLImageProcessor(**parms)
    ori_ret = img_ori_proc([image], return_tensors="pt")
    return ori_ret

test_qwen_images = [
    torch.randint(0, 256, (1920, 1080, 3), dtype=torch.uint8)
]


@pytest.mark.parametrize("qwen_processor_mode", [accdata_processor, transformers_processor])
@pytest.mark.parametrize("input_image", test_qwen_images, ids=["1080p_nhwc"])
@pytest.mark.parametrize("thread_num", [1, 8])
def test_qwen_processor(benchmark, qwen_processor_mode, input_image, thread_num):
    TorchOpTransforms.prepare_torch_thread(thread_num)
    benchmark(qwen_processor_mode, input_image, thread_num)