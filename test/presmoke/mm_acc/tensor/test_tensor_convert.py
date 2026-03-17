import random
import pytest
import numpy as np
from mm import Tensor, TensorFormat, DataType
from mm_test.common import DEVICE_CPU

NP_DTYPE_DICT = {
    DataType.UINT8: np.uint8,
    DataType.INT8: np.int8,
    DataType.FLOAT32: np.float32,
}


def random_ndarray_int32(a: int, b: int) -> np.ndarray:
    arr_int32 = np.array([[random.randint(-2 ** 31, 2 ** 31 - 1) for _ in range(a)] for _ in range(b)], dtype=np.int32)
    return arr_int32


def random_ndarray_fp32(a: int, b: int) -> np.ndarray:
    arr_float32 = np.array([[random.random() for _ in range(a)] for _ in range(b)], dtype=np.float32)
    return arr_float32


def array_full_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return (
            a.shape == b.shape and
            a.dtype == b.dtype and
            a.flags['C_CONTIGUOUS'] == b.flags['C_CONTIGUOUS'] and
            a.flags['F_CONTIGUOUS'] == b.flags['F_CONTIGUOUS'] and
            a.strides == b.strides and
            np.array_equal(a, b)
    )


def array_tensor_equal(a: np.ndarray, b: Tensor):
    assert tuple(b.shape) == a.shape
    assert a.nbytes == b.nbytes
    assert b.device == DEVICE_CPU
    assert b.format == TensorFormat.ND
    assert NP_DTYPE_DICT.get(b.dtype) == a.dtype


def np_to_tensor_to_np(arr: np.ndarray):
    tensor = Tensor.from_numpy(arr)
    array_tensor_equal(arr, tensor)
    arr_new = tensor.numpy()
    assert array_full_equal(arr, arr_new), "Array and tensor data mismatch after conversion cycle"


def test_np_to_tensor_to_np_with_fp32_1x1():
    arr = random_ndarray_fp32(1, 1)
    np_to_tensor_to_np(arr)


def test_tensor_from_numpy_int32():
    arr = random_ndarray_int32(3, 3)
    with pytest.raises(ValueError) as exc_info:
        tensor = Tensor.from_numpy(arr)
    assert "The input numpy's ndarray data type must be in [np.int8/np.uint8/np.float32]" in str(exc_info.value)


def test_tensor_from_numpy_None():
    with pytest.raises(TypeError) as exc_info:
        tensor = Tensor.from_numpy(None)
    assert "The input param 'nd_array' must be of numpy's ndarray type" in str(exc_info.value)