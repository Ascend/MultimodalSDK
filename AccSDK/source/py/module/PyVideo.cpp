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
 * Description: video file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "PyVideo.h"
#include "acc/video/Video.h"
#include "acc/ErrorCode.h"
#include "acc/image/Image.h"

namespace PyAcc {

std::vector<Image> video_decode(const char* path, const char* device, const std::set<uint32_t>& frameIndices,
                                int sampleNum)
{
    std::vector<Acc::Image> accImageResult;
    auto ret = Acc::VideoDecode(path, device, accImageResult, frameIndices, sampleNum);
    if (ret != 0) {
        throw std::runtime_error("Failed to decode video, please see above log for detail.");
    }

    std::vector<Image> result;
    result.reserve(accImageResult.size());
    for (auto& accImage : accImageResult) {
        Image pyImage;
        pyImage.SetImage(accImage);
        result.push_back(std::move(pyImage));
    }
    return result;
}

} // namespace PyAcc
