#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""Presmoke tests for mm.video_info (real _acc extension, no decoding).

``video_info`` reads frame rate / frame count / duration without decoding;
validation rules are identical to ``video_decode`` (mp4 suffix, permission
<= 0640, regular file owned by the current user).
"""

# pylint: disable=duplicate-code
import logging
import os

import pytest

from mm import video_info
from mm_test.common import TEST_HW_USER_VIDEO_PATH

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_PATH = os.path.join(TEST_HW_USER_VIDEO_PATH, "city_1080p_30fps_5min.mp4")


def _require_video():
    if not os.path.isfile(VIDEO_PATH):
        pytest.skip(f"test video not available: {VIDEO_PATH}")


def test_video_info_returns_metadata():
    """video_info returns complete metadata for a real video (no decoding)."""
    _require_video()
    info = video_info(VIDEO_PATH)
    logger.info("video_info: %s", info)
    assert set(info) == {"fps", "n_frames", "duration_sec"}
    assert info["fps"] > 0
    assert info["n_frames"] > 0
    assert info["duration_sec"] > 0
    # consistency: n_frames ≈ fps * duration (within 10% for CFR videos)
    assert abs(info["n_frames"] - info["fps"] * info["duration_sec"]) <= 0.1 * info["n_frames"]


def test_video_info_file_not_exist():
    file_path = os.path.join(TEST_HW_USER_VIDEO_PATH, "nonexistent_video_12345.mp4")
    with pytest.raises(RuntimeError):
        video_info(file_path)


def test_video_info_not_mp4_suffix(tmp_path):
    # TEST_HW_USER_VIDEO_PATH is a read-only dataset mount in CI, so the fixture
    # file must be created in the writable pytest tmp dir instead.
    file_path = str(tmp_path / "not_mp4.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("test\n")
    os.chmod(file_path, 0o640)
    with pytest.raises(RuntimeError):
        video_info(file_path)


def test_video_info_empty_path():
    with pytest.raises(RuntimeError):
        video_info("")


def test_video_info_directory_instead_of_file():
    # TEST_HW_USER_VIDEO_PATH itself is a directory, not a regular file, so it must be rejected
    with pytest.raises(RuntimeError):
        video_info(TEST_HW_USER_VIDEO_PATH)
