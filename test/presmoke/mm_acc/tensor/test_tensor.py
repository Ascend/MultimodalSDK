import pytest
from mm import Tensor, TensorFormat, DataType


def test_tensor_init():
    tensor = Tensor()
    device = 'cpu'
    assert tensor.dtype == DataType.FLOAT32
    assert tensor.device == device
    assert len(tensor.shape) == 0
    assert tensor.format == TensorFormat.ND
    assert tensor.nbytes == 0


def test_tensor_set_format_none():
    tensor = Tensor()
    with pytest.raises(AttributeError) as exc_info:
        tensor.set_format(None)
    assert "'NoneType' object has no attribute 'value'" in str(exc_info.value)


def test_tensor_clone():
    tensor1 = Tensor()
    tensor2 = tensor1.clone()
    assert tensor1.shape == tensor2.shape
    assert tensor1.dtype is tensor2.dtype
    assert tensor1.device == tensor2.device
    assert tensor1.nbytes == tensor2.nbytes
    assert tensor1.format == tensor2.format


def test_tensor_set_format_NHWC():
    tensor1 = Tensor()
    with pytest.raises(RuntimeError) as exc_info:
        tensor1.set_format(TensorFormat.NHWC)
    assert "Tensor set_format failed" in str(exc_info.value)


def test_tensor_set_format_with_clone():
    tensor1 = Tensor()
    tensor1.set_format(TensorFormat.ND)
    assert tensor1.format == TensorFormat.ND
    tensor2 = tensor1.clone()
    assert tensor2.format == TensorFormat.ND