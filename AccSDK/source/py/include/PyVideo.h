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

#include <vector>
#include <set>
#include "PyImage.h"

namespace PyAcc {

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

} // namespace PyAcc

#endif // PYVIDEO_H
