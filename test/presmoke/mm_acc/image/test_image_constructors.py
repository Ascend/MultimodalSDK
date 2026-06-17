import numpy as np
import torch
from PIL import Image as PImage
from mm import Image, ImageFormat
from mm_test.common import DEVICE_CPU, TEST_HW_USER_IMAGES_PATH
import os


def test_from_numpy():
    """Test Image.from_numpy() and verify with PIL."""
    # Create random numpy array
    h, w = 300, 400
    np_arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # Create from numpy
    mm_img = Image.from_numpy(np_arr, ImageFormat.RGB)

    # Create from PIL for comparison
    pil_img = PImage.fromarray(np_arr, mode='RGB')
    mm_img_from_pil = Image.from_pillow(pil_img)

    # Convert both to numpy and compare
    np_mm1 = mm_img.numpy()
    np_mm2 = mm_img_from_pil.numpy()

    assert np.array_equal(np_mm1, np_arr), "Image.from_numpy() result mismatch"
    assert np.array_equal(np_mm2, np_arr), "Image.from_pillow() result mismatch"
    assert np.array_equal(np_mm1, np_mm2), "from_numpy and from_pillow results mismatch"


def test_from_torch():
    """Test Image.from_torch() and verify with PIL."""
    # Create random torch tensor
    h, w = 300, 400
    torch_tensor = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)

    # Create from torch
    mm_img = Image.from_torch(torch_tensor, ImageFormat.RGB)

    # Create from PIL for comparison
    np_arr = torch_tensor.numpy()
    pil_img = PImage.fromarray(np_arr, mode='RGB')
    mm_img_from_pil = Image.from_pillow(pil_img)

    # Convert both to numpy and compare
    np_mm1 = mm_img.numpy()
    np_mm2 = mm_img_from_pil.numpy()

    assert np.array_equal(np_mm1, np_arr), "Image.from_torch() result mismatch"
    assert np.array_equal(np_mm2, np_arr), "Image.from_pillow() result mismatch"
    assert np.array_equal(np_mm1, np_mm2), "from_torch and from_pillow results mismatch"


def test_clone_pixel_values():
    """Test that clone() preserves pixel values, not just metadata."""
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_1024x1920.jpeg")
    image = Image.open(file_path, DEVICE_CPU)

    image_b = image.clone()
    image_c = image_b.clone()

    # Convert all to numpy and compare
    np_original = image.numpy()
    np_b = image_b.numpy()
    np_c = image_c.numpy()

    assert np.array_equal(np_original, np_b), "Clone image B pixel values mismatch"
    assert np.array_equal(np_b, np_c), "Clone image C pixel values mismatch"
    assert np.array_equal(np_original, np_c), "Original and clone C pixel values mismatch"


def test_pillow_conversion():
    """Test round-trip conversion: file -> Image -> PIL -> Image -> numpy."""
    file_path = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_2560x1920.jpeg")

    # Read with Image
    mm_img = Image.open(file_path, DEVICE_CPU)
    np_mm = mm_img.numpy()

    # Convert to PIL
    pil_img = mm_img.pillow()
    assert isinstance(pil_img, PImage.Image), "pillow() should return PIL Image"

    # Convert back to Image
    mm_img_back = Image.from_pillow(pil_img)
    np_mm_back = mm_img_back.numpy()

    # Compare
    assert np.array_equal(np_mm, np_mm_back), "Round-trip conversion pixel values mismatch"


def test_from_numpy_edge_cases():
    """Test edge cases for from_numpy."""
    # Minimal size
    np_small = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.from_numpy(np_small, ImageFormat.RGB)
    assert img.height == 10
    assert img.width == 10

    # Non-contiguous array
    np_non_contiguous = np.random.randint(0, 256, (200, 150, 3), dtype=np.uint8)[:, ::2, :]  # Make non-contiguous
    np_contiguous = np.ascontiguousarray(np_non_contiguous)

    # This should raise because the array is not C-contiguous
    try:
        _ = Image.from_numpy(np_non_contiguous, ImageFormat.RGB)
        assert False, "Should have raised ValueError for non-contiguous array"
    except ValueError:
        pass  # Expected

    # This should work
    img_good = Image.from_numpy(np_contiguous, ImageFormat.RGB)
    np_back = img_good.numpy()
    assert np.array_equal(np_back, np_contiguous)
