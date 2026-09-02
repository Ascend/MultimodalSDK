/*
 * -------------------------------------------------------------------------
 *  This file is part of the MultimodalSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MultimodalSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *           http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 * Description: video head file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef PYVIDEO_H
#define PYVIDEO_H

#include <cstdint>
#include <set>
#include <vector>

#include "PyImage.h"

namespace PyAcc
{

/**
 * @brief Basic metadata of a video file (no decoding involved).
 * @note Field set must stay in sync with Acc::VideoInfo
 *       (AccSDK/include/acc/video/Video.h); the boundary copy in
 *       PyVideo.cpp keeps the internal type out of the Python API,
 *       same pattern as PyImage vs Acc::Image.
 */
struct VideoInfo
{
    double fps;          // average frame rate, frames per second
    int64_t nFrames;     // total frame count
    double durationSec;  // duration in seconds, stream duration preferred, container duration as fallback
};

/**
 * @brief Python interface entry for video decode
 *
 * @param path Input video path
 * @param device video decode device, support "cpu".
 * @param frameIndices expected retained frame indices, range from 0 to max index of video frame.
 * @param sampleNum expected retained frame nums, range from 0 to max nums of video frame, priority lower than
 * frameIndices
 * @return std::vector<Image> Output Image after video decode
 */
std::vector<Image> video_decode(const char* path, const char* device, const std::set<uint32_t>& frameIndices = {},
                                int sampleNum = -1);

/**
 * @brief Python interface entry for video metadata (fps / frame count / duration)
 *
 * @param path Input video path, same validation rules as video_decode (mp4 suffix, permission <= 0640,
 *             not a symbolic link, owned by current user)
 * @return VideoInfo Output video metadata
 */
VideoInfo video_info(const char* path);

}  // namespace PyAcc

#endif  // PYVIDEO_H
