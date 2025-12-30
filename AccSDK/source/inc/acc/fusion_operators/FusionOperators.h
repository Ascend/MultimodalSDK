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
 * Description: Head file for Fusion Operator
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef FUSION_OPERATOR_H
#define FUSION_OPERATOR_H
#include <vector>
#include "acc/image/Image.h"
#include "acc/ErrorCode.h"
#include "acc/core/framework/Pipeline.h"

namespace Acc {

/**
 * @brief Preprocessing configuration for QwenFusion
 *
 * Contains parameters for image normalization and resizing.
 */
struct QwenPreprocessConfig {
    std::vector<float> mean;
    std::vector<float> std;
    int resizeW = 0;
    int resizeH = 0;
};

class FusionOperator {
public:
    /**
     * @brief Preprocess input images with Resize + ToTensor + Normalize
     *
     * @param images List of input images (RGB, 3 channels)
     * @param config Preprocessing parameters including mean, std, resizeW, resizeH, layout
     * @param outputTensors Output tensor list, float32 type, layout current is only NHWC
     * @return ErrorCode Returns SUCCESS if successful, otherwise returns a specific error code
     */
    static ErrorCode Qwen2VLImagePreprocess(const std::vector<std::shared_ptr<Image>>& images,
                                            const QwenPreprocessConfig& config, std::vector<Tensor>& outputTensors);
};

} // namespace Acc

#endif // FUSION_OPERATOR_H