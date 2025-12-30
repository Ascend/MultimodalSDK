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
import unittest
import random
import torch
from torchvision import transforms
import numpy as np

from mm import Tensor, TensorFormat, DataType, DeviceMode, normalize

DEVICE = 'cpu'
DATA_LIST = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
TEST_ARRAY_UINT8 = np.array(DATA_LIST, dtype=np.uint8)
TEST_ARRAY_INT8 = np.array(DATA_LIST, dtype=np.int8)
TEST_ARRAY_FLOAT32 = np.array(DATA_LIST, dtype=np.float32)
TEST_ARRAY_FLOAT16 = np.array(DATA_LIST, dtype=np.float16)

TEST_TORCH_TENSOR_UINT8 = torch.tensor(DATA_LIST, dtype=torch.uint8)
TEST_TORCH_TENSOR_INT8 = torch.tensor(DATA_LIST, dtype=torch.int8)
TEST_TORCH_TENSOR_FLOAT32 = torch.tensor(DATA_LIST, dtype=torch.float32)
TEST_TORCH_TENSOR_FLOAT16 = torch.tensor(DATA_LIST, dtype=torch.float16)
TEST_TORCH_TENSOR_FLOAT32_NHWC = torch.randn(size=(1, 11, 11, 3), dtype=torch.float32)
TEST_TORCH_TENSOR_UINT8_NHWC = torch.randint(high=10, size=(1, 11, 11, 3), dtype=torch.uint8)
TEST_TORCH_TENSOR_FLOAT32_NCHW = torch.randn(size=(1, 3, 11, 11), dtype=torch.float32)

TEST_THREE_CHANNEL = 3
TEST_NORMALIZE_MEAN = [random.uniform(0, 1) for _ in range(TEST_THREE_CHANNEL)]
TEST_NORMALIZE_STD = [random.uniform(0, 1) for _ in range(TEST_THREE_CHANNEL)]


# float: The error between each data point does not exceed one ten-thousandth, and the total number of data points with
# errors exceeding one ten-thousandth does not exceed one ten-thousandth of the total data count.
def compare_float_torch_tensors(tensor, expect_tensor, relative_tol=1e-4, absolute_tol=1e-5, error_tol=1e-4):
    if torch.equal(tensor, expect_tensor):
        return True

    compare_tensor = torch.isclose(tensor, expect_tensor, atol=absolute_tol, rtol=relative_tol)
    if torch.all(compare_tensor):
        return True

    correct_count = float(compare_tensor.sum())
    ele_count = compare_tensor.numel()
    err_ratio = (ele_count - correct_count) / ele_count
    return err_ratio < error_tol


class TestTensor(unittest.TestCase):
    def test_tensor_get_properties_success(self):
        tensor = Tensor()
        device = 'cpu'
        self.assertEqual(tensor.dtype, DataType.FLOAT32)
        self.assertEqual(tensor.device, device)
        self.assertEqual(len(tensor.shape), 0)
        self.assertEqual(tensor.format, TensorFormat.ND)
        self.assertEqual(tensor.nbytes, 0)

    def test_tensor_clone_success(self):
        tensor1 = Tensor()
        tensor2 = tensor1.clone()
        device = 'cpu'
        self.assertEqual(tensor2.dtype, DataType.FLOAT32)
        self.assertEqual(tensor2.device, device)
        self.assertEqual(len(tensor2.shape), 0)
        self.assertEqual(tensor2.format, TensorFormat.ND)
        self.assertEqual(tensor2.nbytes, 0)

    def test_tensor_set_format_success_with_correct_format(self):
        tensor = Tensor()
        tensor.set_format(TensorFormat.ND)
        self.assertEqual(tensor.format, TensorFormat.ND)

    def test_tensor_set_format_fail_with_wrong_format(self):
        tensor = Tensor()
        with self.assertRaises(Exception):
            tensor.set_format(TensorFormat.NHWC)

    def test_numpy_to_tensor_success(self):
        tensor_uint8 = Tensor.from_numpy(TEST_ARRAY_UINT8)
        self.assertEqual(tensor_uint8.dtype, DataType.UINT8)
        self.assertEqual(tensor_uint8.device, DEVICE)
        self.assertEqual(tuple(tensor_uint8.shape), TEST_ARRAY_UINT8.shape)
        self.assertEqual(tensor_uint8.format, TensorFormat.ND)
        self.assertEqual(tensor_uint8.nbytes, TEST_ARRAY_UINT8.nbytes)

        tensor_int8 = Tensor.from_numpy(TEST_ARRAY_INT8)
        self.assertEqual(tensor_int8.dtype, DataType.INT8)
        self.assertEqual(tensor_int8.device, DEVICE)
        self.assertEqual(tuple(tensor_int8.shape), TEST_ARRAY_INT8.shape)
        self.assertEqual(tensor_int8.format, TensorFormat.ND)
        self.assertEqual(tensor_int8.nbytes, TEST_ARRAY_INT8.nbytes)

        tensor_float32 = Tensor.from_numpy(TEST_ARRAY_FLOAT32)
        self.assertEqual(tensor_float32.dtype, DataType.FLOAT32)
        self.assertEqual(tensor_float32.device, DEVICE)
        self.assertEqual(tuple(tensor_float32.shape), TEST_ARRAY_FLOAT32.shape)
        self.assertEqual(tensor_float32.format, TensorFormat.ND)
        self.assertEqual(tensor_float32.nbytes, TEST_ARRAY_FLOAT32.nbytes)

    def test_tensor_to_numpy_success(self):
        tensor_uint8 = Tensor.from_numpy(TEST_ARRAY_UINT8)
        numpy_array_uint8 = tensor_uint8.numpy()

        self.assertEqual(type(numpy_array_uint8), type(TEST_ARRAY_UINT8))
        self.assertEqual(numpy_array_uint8.shape, TEST_ARRAY_UINT8.shape)
        self.assertEqual(numpy_array_uint8.reshape(1, 3, 4).shape, TEST_ARRAY_UINT8.reshape(1, 3, 4).shape)
        self.assertEqual(numpy_array_uint8[:, ::1].any(), TEST_ARRAY_UINT8[:, ::1].any())
        self.assertEqual(np.sum(numpy_array_uint8), np.sum(TEST_ARRAY_UINT8))

        tensor_int8 = Tensor.from_numpy(TEST_ARRAY_INT8)
        numpy_array_int8 = tensor_int8.numpy()

        self.assertEqual(type(numpy_array_int8), type(TEST_ARRAY_INT8))
        self.assertEqual(numpy_array_int8.shape, TEST_ARRAY_INT8.shape)
        self.assertEqual(numpy_array_int8.reshape(1, 3, 4).shape, TEST_ARRAY_INT8.reshape(1, 3, 4).shape)
        self.assertEqual(numpy_array_int8[:, ::1].any(), TEST_ARRAY_INT8[:, ::1].any())
        self.assertEqual(np.sum(numpy_array_int8), np.sum(TEST_ARRAY_INT8))

        tensor_float32 = Tensor.from_numpy(TEST_ARRAY_FLOAT32)
        numpy_array_float32 = tensor_float32.numpy()

        self.assertEqual(type(numpy_array_float32), type(TEST_ARRAY_FLOAT32))
        self.assertEqual(numpy_array_float32.shape, TEST_ARRAY_FLOAT32.shape)
        self.assertEqual(numpy_array_float32.reshape(1, 3, 4).shape, TEST_ARRAY_FLOAT32.reshape(1, 3, 4).shape)
        self.assertEqual(numpy_array_float32[:, ::1].any(), TEST_ARRAY_FLOAT32[:, ::1].any())
        self.assertEqual(np.sum(numpy_array_float32), np.sum(TEST_ARRAY_FLOAT32))

    def test_numpy_to_tensor_fail_with_wrong_input(self):
        with self.assertRaises(TypeError) as context:
            Tensor.from_numpy(TEST_TORCH_TENSOR_UINT8)
        expected_message = "The input param 'nd_array' must be of numpy's ndarray type."
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_input_no_contiguous(self):
        with self.assertRaises(ValueError) as context:
            Tensor.from_numpy(TEST_ARRAY_UINT8.T)
        expected_message = ("The input param 'nd_array' must be c_contiguous. Please use np.ascontiguousarray() "
                            "to convert the array to C-contiguous format before passing it to this function.")
        self.assertEqual(str(context.exception), expected_message)

    def test_numpy_to_tensor_fail_with_wrong_data_type(self):
        with self.assertRaises(ValueError) as context:
            Tensor.from_numpy(TEST_ARRAY_FLOAT16)
        expected_message = "The input numpy's ndarray data type must be in [np.int8/np.uint8/np.float32]"
        self.assertEqual(str(context.exception), expected_message)

    def test_tensor_from_torch_tensor_success(self):
        tensor_uint8 = Tensor.from_torch(TEST_TORCH_TENSOR_UINT8)
        self.assertEqual(tensor_uint8.dtype, DataType.UINT8)
        self.assertEqual(tensor_uint8.device, DEVICE)
        self.assertEqual(tuple(tensor_uint8.shape), TEST_ARRAY_UINT8.shape)
        self.assertEqual(tensor_uint8.format, TensorFormat.ND)
        self.assertEqual(tensor_uint8.nbytes, TEST_ARRAY_UINT8.nbytes)

        tensor_int8 = Tensor.from_torch(TEST_TORCH_TENSOR_INT8)
        self.assertEqual(tensor_int8.dtype, DataType.INT8)
        self.assertEqual(tensor_int8.device, DEVICE)
        self.assertEqual(tuple(tensor_int8.shape), TEST_ARRAY_INT8.shape)
        self.assertEqual(tensor_int8.format, TensorFormat.ND)
        self.assertEqual(tensor_int8.nbytes, TEST_ARRAY_INT8.nbytes)

        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32)
        self.assertEqual(tensor_float32.dtype, DataType.FLOAT32)
        self.assertEqual(tensor_float32.device, DEVICE)
        self.assertEqual(tuple(tensor_float32.shape), TEST_ARRAY_FLOAT32.shape)
        self.assertEqual(tensor_float32.format, TensorFormat.ND)
        self.assertEqual(tensor_float32.nbytes, TEST_ARRAY_FLOAT32.nbytes)

    def test_torch_tensor_to_tensor_fail_with_wrong_input(self):
        with self.assertRaises(TypeError) as context:
            Tensor.from_torch(TEST_ARRAY_UINT8)
        expected_message = "The parameter 'torch_tensor' must be of torch.Tensor type."
        self.assertEqual(str(context.exception), expected_message)

    def test_torch_tensor_to_tensor_fail_with_input_no_contiguous(self):
        with self.assertRaises(ValueError) as context:
            Tensor.from_torch(TEST_TORCH_TENSOR_UINT8.T)
        expected_message = ("The parameter 'torch_tensor' must be c_contiguous. Please "
                             "use torch_tensor.contiguous() to convert the tensor to "
                             "contiguous format before passing it to this function.")
        self.assertEqual(str(context.exception), expected_message)

    def test_torch_tensor_to_tensor_fail_with_wrong_data_type(self):
        with self.assertRaises(ValueError) as context:
            Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT16)
        expected_message = "The input torch_tensor's data type must be in [torch.int8/torch.uint8/torch.float32]"
        self.assertEqual(str(context.exception), expected_message)

    def test_torch_tensor_to_tensor_fail_with_invalid_torch_tensor_device(self):
        mock_tensor = unittest.mock.MagicMock()
        mock_tensor.__class__ = torch.Tensor  # 设置类型
        mock_tensor.device.type = 'npu'
        mock_tensor.is_contiguous.return_value = True
        mock_tensor.dtype = torch.float32
        mock_tensor.numpy.return_value = [[1, 2]]

        with self.assertRaises(ValueError) as context:
            Tensor.from_torch(mock_tensor)
        expected_message = ("The parameter 'torch_tensor' must be on CPU device, "
                             "Please use torch_tensor.cpu() to move the tensor to CPU "
                             "before passing it to this function.")
        self.assertEqual(str(context.exception), expected_message)

    def test_tensor_to_torch_tensor_success(self):
        tensor_uint8 = Tensor.from_torch(TEST_TORCH_TENSOR_UINT8)
        torch_tensor_uint8 = tensor_uint8.torch()

        self.assertEqual(type(torch_tensor_uint8), type(TEST_TORCH_TENSOR_UINT8))
        self.assertEqual(torch_tensor_uint8.shape, TEST_TORCH_TENSOR_UINT8.shape)
        self.assertEqual(torch_tensor_uint8.reshape(1, 3, 4).shape, TEST_TORCH_TENSOR_UINT8.reshape(1, 3, 4).shape)
        self.assertEqual(torch_tensor_uint8[:, ::1].any(), TEST_TORCH_TENSOR_UINT8[:, ::1].any())
        self.assertEqual(torch.sum(torch_tensor_uint8), torch.sum(TEST_TORCH_TENSOR_UINT8))

        tensor_int8 = Tensor.from_torch(TEST_TORCH_TENSOR_INT8)
        torch_tensor_int8 = tensor_int8.torch()

        self.assertEqual(type(torch_tensor_int8), type(TEST_TORCH_TENSOR_INT8))
        self.assertEqual(torch_tensor_int8.shape, TEST_TORCH_TENSOR_INT8.shape)
        self.assertEqual(torch_tensor_int8.reshape(1, 3, 4).shape, TEST_TORCH_TENSOR_INT8.reshape(1, 3, 4).shape)
        self.assertEqual(torch_tensor_int8[:, ::1].any(), TEST_TORCH_TENSOR_INT8[:, ::1].any())
        self.assertEqual(torch.sum(torch_tensor_int8), torch.sum(TEST_TORCH_TENSOR_INT8))

        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32)
        torch_tensor_float32 = tensor_float32.torch()

        self.assertEqual(type(torch_tensor_float32), type(TEST_TORCH_TENSOR_FLOAT32))
        self.assertEqual(torch_tensor_float32.shape, TEST_TORCH_TENSOR_FLOAT32.shape)
        self.assertEqual(torch_tensor_float32.reshape(1, 3, 4).shape, TEST_TORCH_TENSOR_FLOAT32.reshape(1, 3, 4).shape)
        self.assertEqual(torch_tensor_float32[:, ::1].any(), TEST_TORCH_TENSOR_FLOAT32[:, ::1].any())
        self.assertEqual(torch.sum(torch_tensor_float32), torch.sum(TEST_TORCH_TENSOR_FLOAT32))

    def test_tensor_normalize_should_success(self):
        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32_NHWC)
        tensor_float32.set_format(TensorFormat.NHWC)
        normalize_tensor = tensor_float32.normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        normalize_torch_tensor = normalize_tensor.torch()

        torch_tensor_nchw = TEST_TORCH_TENSOR_FLOAT32_NHWC.permute(0, 3, 1, 2)  # NHWC -> NCHW
        torch_tensor_normalize_nchw = transforms.Normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)(torch_tensor_nchw)
        torch_tensor_normalize_nchw = torch_tensor_normalize_nchw.permute(0, 2, 3, 1)  # NCHW -> NHWC

        self.assertTrue(compare_float_torch_tensors(normalize_torch_tensor, torch_tensor_normalize_nchw))

        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32_NCHW)
        tensor_float32.set_format(TensorFormat.NCHW)
        normalize_tensor = tensor_float32.normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        normalize_torch_tensor = normalize_tensor.torch()

        torch_tensor_normalize_nchw = transforms.Normalize(TEST_NORMALIZE_MEAN,
                                                           TEST_NORMALIZE_STD)(TEST_TORCH_TENSOR_FLOAT32_NCHW)
        self.assertTrue(compare_float_torch_tensors(normalize_torch_tensor, torch_tensor_normalize_nchw))

    def test_tensor_normalize_should_failed_with_empty_input(self):
        tensor_float32 = Tensor()
        with self.assertRaises(RuntimeError) as context:
            normalize_tensor = tensor_float32.normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        expected_message = "Failed to execute normalize operator, please ensure your inputs are valid."
        self.assertEqual(str(context.exception), expected_message)

    def test_tensor_normalize_should_failed_with_invalid_input_dtype(self):
        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_UINT8_NHWC)
        with self.assertRaises(RuntimeError) as context:
            normalize_tensor = tensor_float32.normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        expected_message = "Failed to execute normalize operator, please ensure your inputs are valid."
        self.assertEqual(str(context.exception), expected_message)

    def test_tensor_normalize_should_failed_with_invalid_mean_std(self):
        invalid_mean_stds = [
            ([0.1, 0.1], [0.1, 0.1]),
            ([0.1, 0.1, 0.1], [0.1, 0.1]),
            ([0.1, 0.1], [0.1, 0.1, 0.1]),
        ]
        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32_NCHW)
        tensor_float32.set_format(TensorFormat.NCHW)
        for (mean, std) in invalid_mean_stds:
            with self.assertRaises(RuntimeError) as context:
                normalize_tensor = tensor_float32.normalize(mean, std)
            expected_message = "Failed to execute normalize operator, please ensure your inputs are valid."
            self.assertEqual(str(context.exception), expected_message)

    def test_normalize_should_success(self):
        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32_NHWC)
        tensor_float32.set_format(TensorFormat.NHWC)
        dst = normalize(tensor_float32, TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        normalize_torch_tensor = dst.torch()

        torch_tensor_nchw = TEST_TORCH_TENSOR_FLOAT32_NHWC.permute(0, 3, 1, 2)  # NHWC -> NCHW
        torch_tensor_normalize_nchw = transforms.Normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)(torch_tensor_nchw)
        torch_tensor_normalize_nchw = torch_tensor_normalize_nchw.permute(0, 2, 3, 1)  # NCHW -> NHWC

        self.assertTrue(compare_float_torch_tensors(normalize_torch_tensor, torch_tensor_normalize_nchw))

        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32_NCHW)
        tensor_float32.set_format(TensorFormat.NCHW)
        dst = normalize(tensor_float32, TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        normalize_torch_tensor = dst.torch()

        torch_tensor_normalize_nchw = transforms.Normalize(TEST_NORMALIZE_MEAN,
                                                           TEST_NORMALIZE_STD)(TEST_TORCH_TENSOR_FLOAT32_NCHW)
        self.assertTrue(compare_float_torch_tensors(normalize_torch_tensor, torch_tensor_normalize_nchw))

    def test_normalize_should_failed_with_invalid_input_or_output(self):
        with self.assertRaises(ValueError) as context:
            dst = normalize(TEST_TORCH_TENSOR_FLOAT32_NCHW, TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
        expected_message = "The parameter 'src' must be mm.Tensor instance."
        self.assertEqual(str(context.exception), expected_message)

    def test_normalize_should_failed_with_invalid_mean_std(self):
        invalid_mean_stds = [
            ([0.1, 0.1], [0.1, 0.1]),
            ([0.1, 0.1, 0.1], [0.1, 0.1]),
            ([0.1, 0.1], [0.1, 0.1, 0.1]),
        ]
        tensor_float32 = Tensor.from_torch(TEST_TORCH_TENSOR_FLOAT32_NCHW)
        tensor_float32.set_format(TensorFormat.NCHW)
        for (mean, std) in invalid_mean_stds:
            with self.assertRaises(RuntimeError) as context:
                normalize_tensor = normalize(tensor_float32, mean, std)
            expected_message = "Failed to execute normalize operator, please ensure your inputs are valid."
            self.assertEqual(str(context.exception), expected_message)



if __name__ == '__main__':
    unittest.main()
