import random
import torch

from torchvision import transforms
from mm import TensorFormat, normalize

from mm import Image, ImageFormat, TensorFormat
from mm_test.tools import compare_float_torch_tensors

TEST_THREE_CHANNEL = 3
TEST_NORMALIZE_MEAN = [random.uniform(0, 1) for _ in range(TEST_THREE_CHANNEL)]
TEST_NORMALIZE_STD = [random.uniform(0, 1) for _ in range(TEST_THREE_CHANNEL)]


def compare_tensor_normalize_and_torch_normalize(tensor_float32, torch_arr):
    tensor_float32.set_format(TensorFormat.NCHW)
    normalize_tensor = tensor_float32.normalize(TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
    normalize_torch_tensor = normalize_tensor.torch()

    torch_tensor_normalize_nchw = transforms.Normalize(TEST_NORMALIZE_MEAN,
                                                       TEST_NORMALIZE_STD)(torch_arr)
    assert compare_float_torch_tensors(normalize_torch_tensor, torch_tensor_normalize_nchw)


def compare_normalize_and_torch_normalize(tensor_float32, torch_arr):
    tensor_float32.set_format(TensorFormat.NCHW)
    normalize_tensor = normalize(tensor_float32, TEST_NORMALIZE_MEAN, TEST_NORMALIZE_STD)
    normalize_torch_tensor = normalize_tensor.torch()

    torch_tensor_normalize_nchw = transforms.Normalize(TEST_NORMALIZE_MEAN,
                                                       TEST_NORMALIZE_STD)(torch_arr)
    assert compare_float_torch_tensors(normalize_torch_tensor, torch_tensor_normalize_nchw)


def test_normalize_image_to_tensor():
    arr = torch.randint(0, 256, (3, 10, 10), dtype=torch.uint8)
    image = Image.from_torch(arr, ImageFormat.RGB_PLANAR)
    tensor_float32 = image.to_tensor()
    arr_float = tensor_float32.torch()

    compare_normalize_and_torch_normalize(tensor_float32, arr_float)


def test_tensor_normalize_image_to_tensor():
    arr = torch.randint(0, 256, (3, 10, 10), dtype=torch.uint8)
    image = Image.from_torch(arr, ImageFormat.RGB_PLANAR)
    tensor_float32 = image.to_tensor()
    arr_float = tensor_float32.torch()

    compare_tensor_normalize_and_torch_normalize(tensor_float32, arr_float)