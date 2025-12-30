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
 * Description: Video header file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef VIDEO_H
#define VIDEO_H

#include <vector>
#include <set>
#include "acc/image/Image.h"

namespace Acc {
/**
 * @brief: Video decode.
 * @param path: video file path.
 * @param device: video decode device, support "cpu".
 * @param frames: result frames.
 * @param frameIndices: expected retained frame indices, range from 0 to max index of video frame.
 * @param sampleNum: expected retained frame nums, range from 0 to max nums of video frame, priority lower than frameIndices
 */
ErrorCode VideoDecode(const char* path, const char* device, std::vector<Image>& frames,
                      const std::set<uint32_t>& frameIndices = std::set<uint32_t>{}, int sampleNum = -1);
} // namespace Acc

#endif // VIDEO_H