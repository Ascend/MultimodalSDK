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
from typing import Tuple
from .._impl import acc as _acc
from .data_type import DataType, ImageFormat, DeviceMode, Interpolation, TensorFormat
from .tensor_wrapper import Tensor
from .util import ObjectWrapper, _ensure_bytes

_SUPPORT_PILLOW_MODE = "RGB"
_RESIZED_SIZE_LEN = 2


class Image:
    __slots__ = ("_inner", "_pillow_source_ndarray", "_torch_source_ndarray", "_numpy_source_ndarray")

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Image cannot be instantiated directly. Use Image.open().")

    @classmethod
    def open(cls, path: str | bytes, device: str | bytes = b"cpu") -> "Image":
        """Construct Image from a given path and device. Device check will be performed in C++ code


        Args:
            path (str | bytes): given path
            device (str | bytes): only support cpu now

        Returns:
            Image: dst image
        """
        path_bytes = _ensure_bytes(path, "path")
        device_bytes = _ensure_bytes(device, "device")
        acc_img = _acc.Image.open(path_bytes, device_bytes)
        obj = object.__new__(cls)
        obj._inner = acc_img
        return obj

    @classmethod
    def from_numpy(
            cls,
            nd_array: "np.ndarray",
            image_format: ImageFormat,
            device: str | bytes = b"cpu"
    ) -> "Image":
        """
        Create an Image instance from a NumPy array. Device check will be performed in C++ code


        Args:
            nd_array (np.ndarray): Input numpy array containing image data.
            image_format (ImageFormat): Format of the image (e.g., RGB, BGR).
            device (str | bytes, optional): only support cpu now

        Returns:
            Image: Created Image object.
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

        if nd_array.dtype != np.uint8:
            raise ValueError("The input numpy's ndarray data type must be np.uint8")
        device_bytes = _ensure_bytes(device, "device")
        acc_img = _acc.Image.from_numpy(nd_array, image_format.value, device_bytes)

        # return a new image object
        obj = object.__new__(cls)
        obj._inner = acc_img
        obj._numpy_source_ndarray = nd_array
        return obj

    @classmethod
    def from_torch(
            cls,
            torch_tensor: "torch.tensor",
            image_format: ImageFormat,
            device: str | bytes = b"cpu"
    ) -> "Image":
        """Support converting torch.Tensor to Image instances

        Args:
            torch_tensor (torch.Tensor): torch's Tensor, support datatype [uint8]
            image_format (ImageFormat): Format of the image (e.g., RGB, BGR).
            device (str | bytes, optional): Target device, range is  b"cpu".

        Returns:
            Image: dst image
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

        if torch_tensor.dtype != torch.uint8:
            raise ValueError("The input torch tensor data type must be torch.uint8")

        if not torch_tensor.device.type == "cpu":
            raise ValueError("The parameter 'torch_tensor' must be on CPU device, "
                             "Please use torch_tensor.cpu() to move the tensor to CPU "
                             "before passing it to this function.")

        # Convert PyTorch tensor to NumPy ndarray
        # This creates a view of the image data without copying (CPU process solution)
        ndarray = torch_tensor.numpy()

        # Create image from the NumPy array
        image = cls.from_numpy(ndarray, image_format, device)
        image._torch_source_ndarray = ndarray
        return image

    @classmethod
    def from_pillow(cls, pillow_image) -> "Image":
        """Support converting PIL.Image.Image to Image instances

        Args:
            pillow_image (PIL.Image.Image): pillow's Image, support image format [RGB]

        Returns:
            Image: dst image
        """
        try:
            from PIL import Image as PImage
        except ImportError as e:
            raise ImportError(f"Please install pillow firstly, error: {e}") from e

        if not isinstance(pillow_image, PImage.Image):
            raise TypeError("The parameter 'pillow_image' must be of PIL.Image.Image type.")

        if pillow_image.mode != _SUPPORT_PILLOW_MODE:
            raise ValueError("The input pillow's Image mode must be in ['RGB']")
        image_format = ImageFormat.RGB

        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(f"Please install numpy firstly, error: {e}") from e

        # Create Image from the NumPy array which generated by pillow image
        ndarray = np.array(pillow_image)
        image = cls.from_numpy(ndarray, image_format)
        image._pillow_source_ndarray = ndarray
        return image

    @classmethod
    def _from_acc(cls, acc_img: "_acc.Image") -> "Image":
        """
        Construct Image from an existing acc.Image instance.

        Args:
            acc_img (_acc.Image): an instance of the underlying C++ acc.Image

        Returns:
            Image: Wrapped Python Image object
        """
        if not isinstance(acc_img, _acc.Image):
            raise TypeError("acc_img must be an instance of _acc.Image")

        obj = object.__new__(cls)
        obj._inner = acc_img
        return obj

    @property
    def device(self) -> str:
        """Get dtype property

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
    def size(self) -> list:
        """Get size property

        Returns:
            list: Image size
        """
        return list(self._inner.size)

    @property
    def format(self) -> ImageFormat:
        """Get format property

        Returns:
            ImageFormat: range is RGB, BGR, RGB_PLANAR, BGR_PLANAR
        """
        val = self._inner.format
        return ImageFormat(val)

    @property
    def nbytes(self) -> int:
        """Get nbytes property

        Returns:
            int: num of bytes
        """
        return self._inner.nbytes

    @property
    def height(self) -> int:
        """Get height property

        Returns:
            int: Image height
        """
        return self._inner.height

    @property
    def width(self) -> int:
        """Get width property

        Returns:
            int: Image width
        """
        return self._inner.width

    def torch(self):
        """Support converting Image instances to torch.Tensor instance

        Returns:
            torch_tensor: dst torch.Tensor instance
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"Please install torch firstly, error: {e}") from e

        # CPU process solution, current numpy() will return data on npu
        torch_tensor = torch.from_numpy(self.numpy())
        return torch_tensor

    def numpy(self):
        """Support converting Image instances to NumPy ndarray array

        Returns:
            nd_array: dst numpy.ndarray instance
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(f"Please install numpy firstly, error: {e}") from e

        # Get the numpy interface dictionary from the underlying C++ object
        interface_dict = self._inner.numpy()
        return np.asarray(ObjectWrapper(interface_dict))

    def clone(self) -> "Image":
        """Image deep copy to a new image

        Returns:
            Image: dst image
        """
        acc_img = self._inner.clone()
        obj = object.__new__(self.__class__)
        obj._inner = acc_img
        return obj

    def pillow(self):
        """Support converting Image instances to pillow image

        Returns:
            pillow_image: dst PIL.Image.Image instance
        """
        try:
            from PIL import Image as PImage
        except ImportError as e:
            raise ImportError(f"Please install pillow firstly, error: {e}") from e

        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(f"Please install numpy firstly, error: {e}") from e

        img_arr = self.numpy()
        return PImage.fromarray(img_arr, _SUPPORT_PILLOW_MODE)

    def crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        device_mode: DeviceMode = DeviceMode.CPU,
    ) -> "Image":
        """_summary_ Image crop

        Args:
            top (int): _description_ Top boundary position of the crop.
            left (int): _description_ Left boundary position of the crop.
            height (int): _description_ Crop height.
            width (int): _description_ Crop width.
            device_mode (DeviceMode): _description_ The mode for running operator. Default value is CPU.

        Returns:
            Image: _description_ cropped image
        """
        acc_img = self._inner.crop(top, left, height, width, device_mode.value)
        obj = object.__new__(self.__class__)
        obj._inner = acc_img
        return obj

    def resize(
        self,
        size: Tuple[int, int],
        interpolation: Interpolation,
        device_mode: DeviceMode = DeviceMode.CPU,
    ) -> "Image":
        """_summary_ Image resize

        Args:
            size (Tuple[int, int]): _description_ Resized size, which is (width, height)
            interpolation (Interpolation): _description_ Interpolation algorithm.
            device_mode (DeviceMode): _description_ The mode for running operator. Default value is CPU.

        Returns:
            Image: _description_
        """
        if len(size) != _RESIZED_SIZE_LEN:
            raise ValueError("size must be a tuple of (width, height)")
        acc_img = self._inner.resize(
            size[0], size[1], interpolation.value, device_mode.value
        )
        obj = object.__new__(self.__class__)
        obj._inner = acc_img
        return obj

    def to_tensor(self, target_format: TensorFormat = TensorFormat.NCHW,
                  device_mode: DeviceMode = DeviceMode.CPU) -> "Tensor":
        """Image to tensor

        Args:
            target_format (TensorFormat): The format of the target tensor. When it matches the original tensor,
                                          it means no conversion is performed. Supports NHWC, NCHW, and empty values.
            device_mode (DeviceMode):  Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.

        Returns:
            Tensor:
        """
        acc_tensor = self._inner.to_tensor(target_format.value, device_mode.value)
        tensor = Tensor()
        tensor._inner = acc_tensor
        return tensor

    def get_inner(self) -> _acc.Image:
        """_summary_ Image get_inner

        Returns:
            _acc.Image: inner Image (PyAcc)
        """
        return getattr(self, "_inner")
