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
        logger.info(f"{file_path} removed")
    return file_path


def test_decode_not_mp4_mp4(capsys):
    file_path = create_path("not_mp4.mp4")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("test\n")
    os.chmod(file_path, 0o640)
    logger.info(f"{file_path} created")
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