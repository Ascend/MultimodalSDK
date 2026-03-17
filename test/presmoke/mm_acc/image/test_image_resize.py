from functools import singledispatch
import os.path

import numpy as np
from PIL import Image as PImage
from mm import Image, DeviceMode, Interpolation
from mm_test.common import DEVICE_CPU, TEST_HW_USER_IMAGES_PATH, TEST_HW_USER_FILE_PATH


def compare_resize_with_mm_and_pil(image: Image, pil: PImage.Image, size: tuple):
    img_mm_resize = image.resize(size, Interpolation.BICUBIC, DeviceMode.CPU)
    np_mm_resize = img_mm_resize.numpy()
    img_pil_resize = pil.resize(size)
    np_pil_resize = np.array(img_pil_resize)
    assert np.array_equal(np_mm_resize, np_pil_resize), \
        (f"Arrays not equal! shape_mm={np_mm_resize.shape}, shape_pil={np_pil_resize.shape}, "
         f"dtype_mm={np_mm_resize.dtype}, dtype_pil={np_pil_resize.dtype}")


def compare_crop_with_mm_and_pil(image: Image, pil: PImage.Image, top: int, left: int, height: int, width: int):
    img_mm_crop = image.crop(top, left, height, width, DeviceMode.CPU)
    np_mm_crop = img_mm_crop.numpy()
    img_pil_crop = pil.crop((left, top, left + width, top + height))
    np_pil_crop = np.array(img_pil_crop)
    assert np.array_equal(np_mm_crop, np_pil_crop), \
        (f"Arrays not equal! shape_mm={np_mm_crop.shape}, shape_pil={np_pil_crop.shape}, "
         f"dtype_mm={np_mm_crop.dtype}, dtype_pil={np_pil_crop.dtype}")


@singledispatch
def compare_image_resize(file_path: str, size: tuple):
    image = Image.open(file_path, DEVICE_CPU)
    pil = PImage.open(file_path)
    compare_resize_with_mm_and_pil(image, pil, size)


@singledispatch
def compare_image_crop(file_path: str, top: int, left: int, height: int, width: int):
    image = Image.open(file_path, DEVICE_CPU)
    pil = PImage.open(file_path)
    compare_crop_with_mm_and_pil(image, pil, top, left, height, width)


def test_crop_1920x1024_image_to_negative10_0():
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_1920x1024.jpeg")
    compare_image_crop(file_path, 1014, 1910, 10, 10)


def test_resize_dog_jpg_data():
    file_path = os.path.join(TEST_HW_USER_FILE_PATH, "dog_1920_1080.jpg")
    compare_image_resize(file_path, (540, 960))