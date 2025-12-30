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
Description: python tensor test.
Author: ACC SDK
Create: 2025
History: NA
"""
import os
import sys

from accsdk_pytest import BaseTestCase
from acc import (Tensor, TensorFormat_ND, TensorFormat_NHWC, DataType_INT8, DataType_UINT8,
                 DataType_FLOAT32, DeviceMode_CPU)


class ObjectWrapper:
    def __init__(self, interface_dict):
        self.__array_interface__ = interface_dict['__array_interface__']

ARRAY_INTERFACE = "__array_interface__"
TEST_ARRAY_SHAPE = (2, 3)
CPU_DEVICE = b'cpu'

TEST_ARRAY_UINT8 = ObjectWrapper(
    {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|u1", "shape": TEST_ARRAY_SHAPE}})
TEST_ARRAY_INT8 = ObjectWrapper(
    {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|i1", "shape": TEST_ARRAY_SHAPE}})
TEST_ARRAY_FLOAT32 = ObjectWrapper(
    {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": TEST_ARRAY_SHAPE}})
INVALID_ARRAY_SHAPE = [1, 3]
INVALID_ARRAY_INTERFACE_DATA_TYPE = {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f6", "shape": (2,)}}
INVALID_ARRAY_INTERFACE_DATA = {ARRAY_INTERFACE: {"data": 18765104, "typestr": "|f4", "shape": (2,)}}
INVALID_ARRAY_INTERFACE_DATA_ADDRESS = {ARRAY_INTERFACE: {"data": (0, False), "typestr": "|f4", "shape": (2,)}}
INVALID_ARRAY_INTERFACE_NO_SHAPE = {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4"}}
INVALID_ARRAY_INTERFACE_WRONG_SHAPE_TYPE = {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": []}}
INVALID_ARRAY_INTERFACE_SHAPE = {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (-1, 2)}}
INVALID_ARRAY_INTERFACE_NO_TYPE = {ARRAY_INTERFACE: {"data": (18765104, False), "shape": (2,)}}
INVALID_ARRAY_INTERFACE_WRONG_TYPE = {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": 123, "shape": (2,)}}
INVALID_ARRAY_INTERFACE_UNSUPPORT_TYPE = {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f9", "shape": (2,)}}


class TestPyTensor(BaseTestCase):
    def test_tensor_get_properties_success(self):
        tensor = Tensor()
        self.assertEqual(tensor.dtype, 0)
        self.assertEqual(tensor.device, CPU_DEVICE)
        self.assertEqual(tensor.shape.size(), 0)
        self.assertEqual(tensor.format, TensorFormat_ND)
        self.assertEqual(tensor.nbytes, 0)

    def test_tensor_clone_success(self):
        tensor1 = Tensor()
        tensor2 = tensor1.clone()
        self.assertEqual(tensor2.dtype, 0)
        self.assertEqual(tensor2.device, CPU_DEVICE)
        self.assertEqual(tensor2.shape.size(), 0)
        self.assertEqual(tensor2.format, TensorFormat_ND)
        self.assertEqual(tensor2.nbytes, 0)

    def test_tensor_set_format_success_with_correct_format(self):
        tensor = Tensor()
        tensor.set_format(TensorFormat_ND)
        self.assertEqual(tensor.format, TensorFormat_ND)

    def test_tensor_set_format_fail_with_wrong_format(self):
        tensor = Tensor()
        with self.assertRaises(Exception):
            tensor.set_format(TensorFormat_NHWC)

    def test_numpy_to_tensor_success(self):
        tensor_uint8 = Tensor.from_numpy(TEST_ARRAY_UINT8)
        self.assertEqual(tuple(tensor_uint8.shape), TEST_ARRAY_SHAPE)
        self.assertEqual(tensor_uint8.dtype, DataType_UINT8)
        self.assertEqual(tensor_uint8.device, CPU_DEVICE)
        self.assertEqual(tensor_uint8.format, TensorFormat_ND)

        tensor_int8 = Tensor.from_numpy(TEST_ARRAY_INT8)
        self.assertEqual(tuple(tensor_int8.shape), TEST_ARRAY_SHAPE)
        self.assertEqual(tensor_int8.dtype, DataType_INT8)
        self.assertEqual(tensor_int8.device, CPU_DEVICE)
        self.assertEqual(tensor_int8.format, TensorFormat_ND)

        tensor_float32 = Tensor.from_numpy(TEST_ARRAY_FLOAT32)
        self.assertEqual(tuple(tensor_float32.shape), TEST_ARRAY_SHAPE)
        self.assertEqual(tensor_float32.dtype, DataType_FLOAT32)
        self.assertEqual(tensor_float32.device, CPU_DEVICE)
        self.assertEqual(tensor_float32.format, TensorFormat_ND)

    def test_numpy_to_tensor_fail_with_wrong_dtype(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_DATA_TYPE))
        expected_message = "Unsupported python numpy ndarray datatype. Only support int8, uint8, float32."
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_empty_numpy_ndarray(self):
        class InvalidObject:
            pass

        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(InvalidObject())
        expected_message = ("The python numpy ndarray does not have the __array_interface__ dictionary in "
                            "its attributes. Please check whether the passed numpy ndarray is corrupted.")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_wrong_numpy_ndarray_data(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_DATA))
        expected_message = ("Invalid data field in __array_interface__ of python numpy ndarray, "
                            "It shouldn't be missing and contain a 2-tuple (address, read-only flag).")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_wrong_numpy_ndarray_data_address(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_DATA_ADDRESS))
        expected_message = ("Failed to get valid data pointer from __array_interface__ of python numpy "
                            "ndarray. The data field's address must be legal")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_numpy_ndarray_no_shape(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_NO_SHAPE))
        expected_message = ("Invalid shape field in __array_interface__ of python numpy ndarray. "
                            "It shouldn't be missing and shouldn't be empty tuple.")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_numpy_ndarray_wrong_shape_type(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_WRONG_SHAPE_TYPE))
        expected_message = ("Invalid shape field in __array_interface__ of python numpy ndarray. "
                            "It shouldn't be missing and shouldn't be empty tuple.")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_wrong_numpy_ndarray_shape(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_SHAPE))
        expected_message = ("Invalid dimension in shape of __array_interface__ of python numpy ndarray. "
                            "The dimension's value must greater than 0.")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_numpy_ndarray_no_type(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_NO_TYPE))
        expected_message = ("Invalid typestr field in __array_interface__ of python numpy ndarray. "
                            "It shouldn't be missing and must be a Unicode object.")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_numpy_ndarray_wrong_type(self):
        with self.assertRaises(RuntimeError) as context:
            Tensor.from_numpy(ObjectWrapper(INVALID_ARRAY_INTERFACE_WRONG_TYPE))
        expected_message = ("Invalid typestr field in __array_interface__ of python numpy ndarray. "
                            "It shouldn't be missing and must be a Unicode object.")
        self.assertEqual(str(context.exception), expected_message)

    def test_tensor_to_numpy_success(self):
        tensor_uint8 = Tensor.from_numpy(TEST_ARRAY_UINT8)
        i_arr_uint8 = tensor_uint8.numpy().get(ARRAY_INTERFACE)
        self.assertEqual(i_arr_uint8["data"][0], TEST_ARRAY_UINT8.__array_interface__["data"][0])
        self.assertEqual(i_arr_uint8["shape"], TEST_ARRAY_UINT8.__array_interface__["shape"])
        self.assertEqual(i_arr_uint8["typestr"], TEST_ARRAY_UINT8.__array_interface__["typestr"])

        tensor_int8 = Tensor.from_numpy(TEST_ARRAY_INT8)
        i_arr_int8 = tensor_int8.numpy().get(ARRAY_INTERFACE)
        self.assertEqual(i_arr_int8["data"][0], TEST_ARRAY_INT8.__array_interface__["data"][0])
        self.assertEqual(i_arr_int8["shape"], TEST_ARRAY_INT8.__array_interface__["shape"])
        self.assertEqual(i_arr_int8["typestr"], TEST_ARRAY_INT8.__array_interface__["typestr"])

        tensor_float32 = Tensor.from_numpy(TEST_ARRAY_FLOAT32)
        i_arr_float32 = tensor_float32.numpy().get(ARRAY_INTERFACE)
        self.assertEqual(i_arr_float32["data"][0], TEST_ARRAY_FLOAT32.__array_interface__["data"][0])
        self.assertEqual(i_arr_float32["shape"], TEST_ARRAY_FLOAT32.__array_interface__["shape"])
        self.assertEqual(i_arr_float32["typestr"], "<f4") # Little-endian

    def test_tensor_normalize_failed_with_invalid_inputs(self):
        invalid_arrs = [
            # invalid dtype
            ObjectWrapper({ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|u1", "shape": (1, 11, 11, 3)}}),
            # invalid batch
            ObjectWrapper({ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (2, 11, 11, 3)}}),
            # invalid channel
            ObjectWrapper({ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (1, 11, 11, 1)}}),
        ]
        mean, std = [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]
        for arr in invalid_arrs:
            src = Tensor.from_numpy(arr)
            src.set_format(TensorFormat_NHWC)
            dst = Tensor()
            with self.assertRaises(RuntimeError) as context:
                dst = src.normalize(mean, std, DeviceMode_CPU)
            expected_message = ("Failed to execute normalize operator, please ensure your inputs are valid.")
            self.assertEqual(str(context.exception), expected_message)

    def test_tensor_normalize_failed_with_invalid_mean_or_std(self):
        invalid_arr = ObjectWrapper(
            {ARRAY_INTERFACE: {"data": (18765104, False), "typestr": "|f4", "shape": (1, 11, 11, 3)}})
        invalid_mean_stds = [
            # invalid mean and std's arr length
            ([0.1, 0.1], [0.1, 0.1]),
            # mean and std's arr length is not equal
            ([0.1, 0.1], [0.1, 0.1, 0.1]),
            ([0.1, 0.1, 0.1], [0.1, 0.1])
        ]
        for (mean, std) in invalid_mean_stds:
            src = Tensor.from_numpy(invalid_arr)
            src.set_format(TensorFormat_NHWC)
            with self.assertRaises(RuntimeError) as context:
                dst = src.normalize(mean, std, DeviceMode_CPU)
            expected_message = ("Failed to execute normalize operator, please ensure your inputs are valid.")
            self.assertEqual(str(context.exception), expected_message)

if __name__ == '__main__':
    failed = TestPyTensor.run_tests()
    sys.exit(1 if failed > 0 else 0)
