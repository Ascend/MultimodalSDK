import numpy as np
import pytest
from PIL import Image as PImage
from torchvision import transforms

from mm import Image, TensorFormat
from mm_test.tools import compare_float_torch_tensors

SUPPORT_MODE = "RGB"
IM_WIDTH = 10
IM_HEIGHT = 10
THREE_CHANNEL = 3


def get_image_and_torch_tensor():
    arr = np.random.randint(0, 255, (IM_HEIGHT, IM_WIDTH, THREE_CHANNEL), dtype=np.uint8)
    pillow_image = PImage.fromarray(arr, mode=SUPPORT_MODE)
    image = Image.from_pillow(pillow_image)
    torchvision_torch_tensor = transforms.ToTensor()(pillow_image)
    return image, torchvision_torch_tensor


def test_10_random_data_image_to_tensor():
    for _ in range(10):
        image, torchvision_torch_tensor = get_image_and_torch_tensor()

        tensor = image.to_tensor()
        mm_torch_tensor = tensor.torch()

        assert compare_float_torch_tensors(mm_torch_tensor, torchvision_torch_tensor.unsqueeze(0))


def test_image_to_NHWC_tensor():
    image, torchvision_torch_tensor = get_image_and_torch_tensor()

    tensor = image.to_tensor(TensorFormat.NHWC)
    mm_torch_tensor = tensor.torch()

    torchvision_torch_tensor = torchvision_torch_tensor.permute(1, 2, 0)
    assert compare_float_torch_tensors(mm_torch_tensor, torchvision_torch_tensor.unsqueeze(0))


def test_image_to_tensor_target_format_is_none():
    image, _ = get_image_and_torch_tensor()
    with pytest.raises(AttributeError) as exc_info:
        image.to_tensor(target_format=None)
    assert "'NoneType' object has no attribute 'value'" in str(exc_info.value)