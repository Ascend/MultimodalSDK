import os.path

import pytest
from mm import Image
from mm_test.common import DEVICE_CPU, TEST_HW_USER_IMAGES_PATH


def test_image_init():
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_1024x10.jpg")
    image = Image.open(file_path, DEVICE_CPU)
    assert list(image.size) == [1024, 10]
    assert image.width == 1024
    assert image.height == 10
    assert image.format.value == 12
    assert image.dtype.value == 4
    assert image.device == DEVICE_CPU


def clone_and_compare(image: Image):
    image_b = image.clone()
    image_c = image_b.clone()
    assert list(image.size) == list(image_b.size) == list(image_c.size)
    assert image.format.value == image_b.format.value == image_c.format.value
    assert image.dtype.value == image_b.dtype.value == image_c.dtype.value
    assert image.device == image_b.device == image_c.device


def test_image_2560x1920_jpeg():
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_2560x1920.jpeg")
    image = Image.open(file_path, DEVICE_CPU)
    clone_and_compare(image)


def test_image_not_exist(capsys):
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_not_exist.jpeg")
    with pytest.raises(RuntimeError) as exc_info:
        Image.open(file_path, DEVICE_CPU)
    assert "Failed to allocate memory" in str(exc_info.value)
    captured = capsys.readouterr()
    assert ("CheckFilePath: Check file path failed. The file does not exist" in captured.out and
            'Code = 0x10100001, Message = "Invalid parameter"' in captured.out)