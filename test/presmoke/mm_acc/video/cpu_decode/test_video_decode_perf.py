import os.path
import time
import logging
import pytest
from mm import video_decode
from mm_test.common import DEVICE_CPU, TEST_HW_USER_VIDEO_PATH

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cpu_video_decode_perf(video_name, frame_id, frame_num, repeat=10):
    file_path = os.path.join(TEST_HW_USER_VIDEO_PATH, video_name)
    durations = []
    frame_total = []
    
    # Warm up
    video_decode(file_path, DEVICE_CPU, frame_id, frame_num)
    
    for _ in range(repeat):
        start = time.time()
        mm_images = video_decode(file_path, DEVICE_CPU, frame_id, frame_num)
        end = time.time()
        durations.append(end - start)
        frame_total.append(len(mm_images))
    
    average_delay = sum(durations) / repeat
    frame_single = sum(frame_total) / repeat
    
    logger.info(f"Durations list: {durations}")
    logger.info(f"Video decode performance - File: {file_path}, Repeat: {repeat}, "
              f"Cumulative Frames: {sum(frame_total)}, Average Frames per Run: {frame_single}, "
              f"Average Latency: {average_delay:.6f} s")
    
    return average_delay


def test_480p_10fps_5min_600_num():
    video_name = '480p_10fps_300s.mp4'
    frame_id = []
    frame_num = 600
    delay = cpu_video_decode_perf(video_name=video_name, frame_id=frame_id, frame_num=frame_num, repeat=2)
    assert delay < (0.475579 * 1.1)


def test_480p_60fps_5min_32_num():
    video_name = '480p_60fps_300s.mp4'
    frame_id = []
    frame_num = 32
    delay = cpu_video_decode_perf(video_name=video_name, frame_id=frame_id, frame_num=frame_num, repeat=3)
    assert delay < (0.415534 * 1.1)