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
from .._impl import acc as _acc
from .data_type import DataType, TensorFormat, DeviceMode
from .util import ObjectWrapper


class Tensor:
    __slots__ = ("_inner",)

    def __init__(self, *args, **kwargs):
        self._inner = _acc.Tensor()

    @classmethod
    def from_numpy(cls, nd_array):
        """Support converting NumPy ndarray array to Tensor instances

        Args:
            nd_array (numpy.ndarray): numpy's ndarray, support datatype [int8/uint8/float32]

        Returns:
            Tensor: dst tensor
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(f"Please install numpy firstly, error: {e}") from e

        if not isinstance(nd_array, np.ndarray):
            raise TypeError("The input param 'nd_array' must be of numpy's ndarray type.")

        if not nd_array.flags['C_CONTIGUOUS']:
            raise ValueError("The input param 'nd_array' must be c_contiguous. Please use np.ascontiguousarray() "
                             "to convert the array to C-contiguous format before passing it to this function.")

        if nd_array.dtype not in (np.int8, np.uint8, np.float32):
            raise ValueError("The input numpy's ndarray data type must be in [np.int8/np.uint8/np.float32]")

        # If an error occurs, an exception will be raised.
        acc_tensor = _acc.Tensor.from_numpy(nd_array)

        # return a new tensor object
        obj = object.__new__(cls)
        obj._inner = acc_tensor
        return obj

    @classmethod
    def from_torch(cls, torch_tensor):
        """Support converting torch.Tensor to Tensor instances

        Args:
            torch_tensor (torch.Tensor): torch's Tensor, support datatype [int8/uint8/float32]

        Returns:
            Tensor: dst tensor
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"Please install torch firstly, error: {e}") from e

        if not isinstance(torch_tensor, torch.Tensor):
            raise TypeError("The parameter 'torch_tensor' must be of torch.Tensor type.")

        if not torch_tensor.is_contiguous():
            raise ValueError("The parameter 'torch_tensor' must be c_contiguous. Please "
                             "use torch_tensor.contiguous() to convert the tensor to "
                             "contiguous format before passing it to this function.")

        if torch_tensor.dtype not in (torch.int8, torch.uint8, torch.float32):
            raise ValueError("The input torch_tensor's data type must be in [torch.int8/torch.uint8/torch.float32]")

        if torch_tensor.device.type != 'cpu':
            raise ValueError("The parameter 'torch_tensor' must be on CPU device, "
                             "Please use torch_tensor.cpu() to move the tensor to CPU "
                             "before passing it to this function.")

        # Convert PyTorch tensor to NumPy ndarray
        # This creates a view of the tensor data without copying (CPU process solution)
        ndarray = torch_tensor.numpy()

        # Create Tensor from the NumPy array
        tensor = cls.from_numpy(ndarray)
        return tensor

    @property
    def device(self) -> str:
        """Get device property

        Returns:
            str: device str
        """
        return self._inner.device.decode("utf-8")

    @property
    def dtype(self) -> DataType:
        """Get dtype property

        Returns:
            DataType: data type
        """
        val = self._inner.dtype
        return DataType(val)

    @property
    def shape(self) -> list:
        """Get shape property

        Returns:
            list: tensor shape
        """
        return list(self._inner.shape)

    @property
    def format(self) -> TensorFormat:
        """Get format property

        Returns:
            TensorFormat: tensor format
        """
        val = self._inner.format
        return TensorFormat(val)

    @property
    def nbytes(self) -> int:
        """Get nbytes property

        Returns:
            int: num of bytes
        """
        return self._inner.nbytes

    def clone(self) -> "Tensor":
        """Tensor deep copy to a new tensor

        Returns:
            Tensor: dst tensor
        """
        acc_tensor = self._inner.clone()
        obj = object.__new__(self.__class__)
        obj._inner = acc_tensor
        return obj

    def set_format(self, tensor_format: TensorFormat):
        """Set format property, range is ND, NHWC, NCHW

        Args:
            format (TensorFormat): range is ND, NHWC, NCHW
        """
        self._inner.set_format(tensor_format.value)

    def numpy(self):
        """Support converting Tensor instances to NumPy ndarray array

        Returns:
            nd_array: dst numpy.ndarray instance
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(f"Please install numpy firstly, error: {e}") from e

        # Get the numpy __array_interface__ dictionary from the underlying C++ object
        # If an error occurs, an exception will be raised.
        interface_dict = self._inner.numpy()
        return np.asarray(ObjectWrapper(interface_dict))

    def torch(self):
        """Support converting Tensor instances to torch.Tensor instance

        Returns:
            torch_tensor: dst torch.Tensor instance
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"Please install torch firstly, error: {e}") from e

        # Convert tensor to CPU, then to numpy array, and finally back to torch tensor
        # This ensures the tensor is processed on CPU (CPU process solution)
        return torch.from_numpy(self.numpy())

    def normalize(self, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU):
        """Normalize the input tensor with given mean and standard deviation.

        Args:
            mean (list[float]): List of mean values for normalization, one value per channel.
            std (list[float]): List of standard deviation values for normalization, one value per channel.
            device_mode (DeviceMode): Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.
        Returns:
            Tensor: dst tensor
        """
        acc_tensor = self._inner.normalize(mean, std, device_mode.value)
        obj = object.__new__(self.__class__)
        obj._inner = acc_tensor

        return obj


def normalize(src: Tensor, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU):
    """Normalize the input tensor with given mean and standard deviation.

    Args:
        src (Tensor): Input tensor to be normalized.
        mean (list[float]): List of mean values for normalization, one value per channel.
        std (list[float]): List of standard deviation values for normalization, one value per channel.
        device_mode (DeviceMode): Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.
    """
    if not isinstance(src, Tensor):
        raise ValueError("The parameter 'src' must be mm.Tensor instance.")

    src_acc_tensor = src._inner
    dst_acc_tensor = _acc.Tensor()
    _acc.normalize(src_acc_tensor, dst_acc_tensor, mean, std, device_mode.value)
    obj = object.__new__(Tensor)
    obj._inner = dst_acc_tensor
    return obj