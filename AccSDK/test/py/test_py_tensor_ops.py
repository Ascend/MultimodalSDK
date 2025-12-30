#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""
Description: python device test.
Author: ACC SDK
Create: 2025
History: NA
"""
import sys

from accsdk_pytest import BaseTestCase
from acc import Tensor, DeviceMode_CPU, DataType_FLOAT32, TensorFormat_NHWC, normalize
from test_py_tensor import ARRAY_INTERFACE, ObjectWrapper


class TestPyTensorOps(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_arr = ObjectWrapper(
            {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (1, 11, 11, 3)}})
        cls.mean = [0.1, 0.1, 0.1]
        cls.std = [0.1, 0.1, 0.1]

    def test_tensor_normalize_failed_with_invalid_inputs(self):
        invalid_arrs = [
            # invalid dtype
            ObjectWrapper({ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|u1", "shape": (1, 11, 11, 3)}}),
            # invalid batch
            ObjectWrapper({ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (2, 11, 11, 3)}}),
            # invalid channel
            ObjectWrapper({ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (1, 11, 11, 1)}}),
        ]
        for arr in invalid_arrs:
            src = Tensor.from_numpy(arr)
            src.set_format(TensorFormat_NHWC)
            dst = Tensor()
            with self.assertRaises(RuntimeError) as context:
                normalize(src, dst, self.mean, self.std, DeviceMode_CPU)
            expected_message = ("Failed to execute normalize operator, please ensure your inputs are valid.")
            self.assertEqual(str(context.exception), expected_message)

    def test_tensor_normalize_failed_with_invalid_mean_or_std(self):
        invalid_mean_stds = [
            # invalid mean and std's arr length
            ([0.1, 0.1], [0.1, 0.1]),
            # mean and std's arr length is not equal
            ([0.1, 0.1], self.std),
            (self.mean, [0.1, 0.1])
        ]
        for (mean, std) in invalid_mean_stds:
            src = Tensor.from_numpy(self.valid_arr)
            src.set_format(TensorFormat_NHWC)
            dst = Tensor()
            with self.assertRaises(RuntimeError) as context:
                normalize(src, dst, mean, std, DeviceMode_CPU)
            expected_message = ("Failed to execute normalize operator, please ensure your inputs are valid.")
            self.assertEqual(str(context.exception), expected_message)

if __name__ == '__main__':
    failed = TestPyTensorOps.run_tests()
    sys.exit(1 if failed > 0 else 0)