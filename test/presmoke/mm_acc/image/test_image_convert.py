import os.path

import numpy as np
import torch
from PIL import Image as PImage
import pytest
from mm import Image
from mm_test.common import DEVICE_CPU, TEST_HW_USER_IMAGES_PATH

IM_WIDTH = 10
IM_HEIGHT = 10
THREE_CHANNEL = 3


def random_hw():
    H = np.random.randint(10, 8193)
    W = np.random.randint(10, 8193)
    return H, W


def compare_img_and_np(image: Image, arr: np.ndarray):
    assert arr.shape == (image.height, image.width, 3)
    assert image.dtype.value == 4
    assert image.device == DEVICE_CPU
    assert arr.dtype == np.uint8


def compare_img_and_torch(image: Image, arr: torch.Tensor):
    assert arr.shape == (image.height, image.width, 3)
    assert image.dtype.value == 4
    assert image.device == DEVICE_CPU
    assert arr.dtype == torch.uint8
    assert arr.device == torch.device("cpu")


def compare_img_and_pil(image: Image, arr: PImage.Image):
    assert arr.size == (image.width, image.height)
    assert image.dtype.value == 4
    assert image.device == DEVICE_CPU
    assert np.array(arr).dtype == np.uint8


def test_convert_1024x1920_image_to_np():
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_1024x1920.jpeg")
    image = Image.open(file_path, DEVICE_CPU)
    compare_img_and_np(image, image.numpy())


def test_convert_1024x1920_pil_to_image():
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_1024x1920.jpeg")
    p_image = PImage.open(file_path)
    image = Image.from_pillow(p_image)
    compare_img_and_pil(image, p_image)