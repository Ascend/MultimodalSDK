import os
import logging
import pytest
from mm import video_decode
from mm_test.common import DEVICE_CPU, TEST_HW_USER_VIDEO_PATH

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_path(file_name):
    file_path = os.path.join(TEST_HW_USER_VIDEO_PATH, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        msg = f"{file_path} removed"
        logger.info(msg)
    return file_path


def test_decode_not_mp4_mp4(capsys):
    file_path = create_path("not_mp4.mp4")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("test\n")
    os.chmod(file_path, 0o640)
    msg = f"{file_path} created"
    logger.info(msg)
    with pytest.raises(RuntimeError) as exc_info:
        video_decode(file_path, DEVICE_CPU, [], 32)
    assert "Failed to decode video, please see above log for detail" in str(exc_info.value)
    captured = capsys.readouterr()
    assert 'Cannot open video file, please ensure that the input video is legal' in captured.out


def test_decode_invalid_device(capsys):
    file_path = os.path.join(TEST_HW_USER_VIDEO_PATH, "city_1080p_30fps_5min.mp4")
    device = 'device'
    with pytest.raises(RuntimeError) as exc_info:
        video_decode(file_path, device, [], 32)
    assert "Failed to decode video, please see above log for detail" in str(exc_info.value)
    captured = capsys.readouterr()
    assert """Only 'cpu' is supported now. (Code = 0x10100002, Message = "Unsupported type")""" in captured.out


def test_decode_file_not_exist(capsys):
    """Test decoding non-existent file."""
    file_path = os.path.join(TEST_HW_USER_VIDEO_PATH, "nonexistent_video_12345.mp4")
    with pytest.raises(RuntimeError) as exc_info:
        video_decode(file_path, DEVICE_CPU, [], 32)
    assert "Failed to decode video, please see above log for detail" in str(exc_info.value)
    captured = capsys.readouterr()
    # Should log file path error
    assert "CheckFilePath" in captured.out or "Cannot open" in captured.out


def test_decode_empty_string_path(capsys):
    """Test decoding with empty string as path."""
    with pytest.raises(RuntimeError) as exc_info:
        video_decode("", DEVICE_CPU, [], 32)
    assert "Failed to decode video, please see above log for detail" in str(exc_info.value)
    captured = capsys.readouterr()
    assert "Invalid parameter" in captured.out


def test_decode_directory_instead_of_file(capsys):
    """Test decoding a directory instead of a file."""
    dir_path = os.path.dirname(os.path.join(TEST_HW_USER_VIDEO_PATH, "dummy"))
    with pytest.raises(RuntimeError) as exc_info:
        video_decode(dir_path, DEVICE_CPU, [], 32)
    assert "Failed to decode video, please see above log for detail" in str(exc_info.value)
    captured = capsys.readouterr()
    assert "Check file path failed. The file is not a regular file" in captured.out


def test_decode_negative_sample_num(capsys):
    """Test decoding with negative sample_num."""
    file_path = os.path.join(TEST_HW_USER_VIDEO_PATH, "city_1080p_30fps_5min.mp4")
    with pytest.raises(RuntimeError) as exc_info:
        video_decode(file_path, DEVICE_CPU, [], -100)
    assert "Failed to decode video, please see above log for detail" in str(exc_info.value)
    captured = capsys.readouterr()
    assert "the target frame number must be less than 9000 and greater than 1" in captured.out
