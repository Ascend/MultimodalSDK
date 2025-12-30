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

from accdata.plugin.pytorch import to_accdata_tensorlist
from accdata.plugin.pytorch import to_torch_tensorlist
import accdata.backend as _b
import accdata.types as _t
import numpy as np


@pytest.mark.smoke
class PluginTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.height = 1024
        self.width = 1024
        self.hwc_shape = (self.batch_size, self.height, self.width, 3)
        self.chw_shape = (self.batch_size, 3, self.height, self.width)
        self.data_nhwc_f32 = np.random.uniform(0, 256, size=self.hwc_shape).astype(np.float32)
        self.data_nchw_f32 = np.random.uniform(0, 256, size=self.chw_shape).astype(np.float32)
        self.data_nchw_uint8 = np.random.uniform(0, 256, size=self.chw_shape).astype(np.uint8)
        self.data_nhwc_uint8 = np.random.uniform(0, 256, size=self.hwc_shape).astype(np.uint8)

    def test_to_accdata_tensorlist_normal_nhwc_f32(self):   # 测试转为accdata tensorlist,要求转换后指向同一内存地址,下同
        torchtl = [torch.from_numpy(self.data_nhwc_f32)]
        actl = to_accdata_tensorlist(torchtl)
        self.assertEqual(actl[0].RawDataPtr(), self.data_nhwc_f32.ctypes.data)

    def test_to_accdata_tensorlist_normal_nchw_f32(self):
        torchtl = [torch.from_numpy(self.data_nchw_f32)]
        actl = to_accdata_tensorlist(torchtl)
        self.assertEqual(actl[0].RawDataPtr(), self.data_nchw_f32.ctypes.data)

    def test_to_accdata_tensorlist_normal_nhwc_uint8(self):
        torchtl = [torch.from_numpy(self.data_nhwc_uint8)]
        actl = to_accdata_tensorlist(torchtl)
        self.assertEqual(actl[0].RawDataPtr(), self.data_nhwc_uint8.ctypes.data)

    def test_to_accdata_tensorlist_normal_nchw_uint8(self):
        torchtl = [torch.from_numpy(self.data_nchw_uint8)]
        actl = to_accdata_tensorlist(torchtl)
        self.assertEqual(actl[0].RawDataPtr(), self.data_nchw_uint8.ctypes.data)

    def test_to_accdata_tensorlist_invlid_layout(self):     # 测试不支持的shape场景
        _data = np.random.uniform(0, 256, size=(1, 2, 2, 2)).astype(np.float32)
        torchtl = [torch.from_numpy(_data)]
        with self.assertRaises(TypeError) as context:
            actl = to_accdata_tensorlist(torchtl)

    def test_to_accdata_tensorlist_wrong_tensor_type(self):   # 测试不支持的tensor数据类型
        with self.assertRaises(TypeError) as context:
            tttl = to_accdata_tensorlist([1, 3, 4])
        with self.assertRaises(TypeError) as context:
            tttl = to_accdata_tensorlist(1)
        with self.assertRaises(TypeError) as context:
            tttl = to_accdata_tensorlist(None)
        with self.assertRaises(TypeError) as context:
            tttl = to_accdata_tensorlist("hello")
        with self.assertRaises(TypeError) as context:
            tttl = to_accdata_tensorlist(True)

    def test_to_torch_tensorlist_normal_nhwc_f32(self):     # 测试转为torch tensorlist,要求转换后指向同一内存地址,下同
        actl = _b.new_tensorlist(1)
        actl[0].ShareData(self.data_nhwc_f32, _t.TensorLayout.NHWC)
        tttl = to_torch_tensorlist(actl)
        self.assertEqual(tttl[0].data_ptr(), self.data_nhwc_f32.ctypes.data)

    def test_to_torch_tensorlist_normal_nchw_f32(self):
        actl = _b.new_tensorlist(1)
        actl[0].ShareData(self.data_nchw_f32, _t.TensorLayout.NCHW)
        tttl = to_torch_tensorlist(actl)
        self.assertEqual(tttl[0].data_ptr(), self.data_nchw_f32.ctypes.data)

    def test_to_torch_tensorlist_normal_nhwc_uint8(self):
        actl = _b.new_tensorlist(1)
        actl[0].ShareData(self.data_nhwc_uint8, _t.TensorLayout.NHWC)
        tttl = to_torch_tensorlist(actl)
        self.assertEqual(tttl[0].data_ptr(), self.data_nhwc_uint8.ctypes.data)

    def test_to_torch_tensorlist_normal_nchw_uint8(self):
        actl = _b.new_tensorlist(1)
        actl[0].ShareData(self.data_nchw_uint8, _t.TensorLayout.NCHW)
        tttl = to_torch_tensorlist(actl)
        self.assertEqual(tttl[0].data_ptr(), self.data_nchw_uint8.ctypes.data)

    def test_to_torch_tensorlist_wrong_tensor_type(self):   # 测试不支持的tensor数据类型
        with self.assertRaises(TypeError) as context:
            tttl = to_torch_tensorlist(torch.from_numpy(self.data_nchw_f32))
        with self.assertRaises(TypeError) as context:
            tttl = to_torch_tensorlist(1)
        with self.assertRaises(TypeError) as context:
            tttl = to_torch_tensorlist(None)
        with self.assertRaises(TypeError) as context:
            tttl = to_torch_tensorlist("hello")
        with self.assertRaises(TypeError) as context:
            tttl = to_torch_tensorlist(True)

if __name__ == "__main__":
    unittest.main()