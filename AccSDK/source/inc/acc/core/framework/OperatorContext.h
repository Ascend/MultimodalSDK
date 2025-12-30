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
 * Description: Operator's context for checker and function.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef OPERATOR_CONTEXT_H
#define OPERATOR_CONTEXT_H

#include <vector>

#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorDataType.h"

namespace Acc {
    // Operator validation context type
    struct OperatorContext {
        const std::vector<std::reference_wrapper<const Tensor>> inputTensorRefs;
        const std::vector<std::reference_wrapper<Tensor>> outputTensorRefs;
        /**
         * @brief Destroy the Operator Context object
         *
         */
        virtual ~OperatorContext() = default;
        /**
         * @brief Construct a new Operator Context object
         *
         * @param inputTensorRefs vector of input tensors reference
         * @param outputTensorRefs vector of output tensors reference
         */
        OperatorContext(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                        const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs)
            : inputTensorRefs(inputTensorRefs), outputTensorRefs(outputTensorRefs)
        {
        }
    };

    struct ResizeContext : OperatorContext {
        size_t resizedH; // target resize height
        size_t resizedW; // target resize weight
        Interpolation interpolation;
        DeviceMode deviceMode;
        ResizeContext(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                      const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs, size_t resizedH,
                      size_t resizedW, const Interpolation interpolation, DeviceMode deviceMode)
            : OperatorContext(inputTensorRefs, outputTensorRefs),
              resizedH(resizedH),
              resizedW(resizedW),
              interpolation(interpolation),
              deviceMode(deviceMode)
        {
        }
    };

    struct CropContext : OperatorContext {
        uint32_t top;    // Top starting position of the crop region (Y coordinate)
        uint32_t left;   // Left starting position of the crop region (X coordinate)
        uint32_t height; // Height of the crop region (number of pixels from top downward)
        uint32_t width;  // Width of the crop region (number of pixels from left rightward)
        DeviceMode deviceMode;
        CropContext(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                    const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs, uint32_t top, uint32_t left,
                    uint32_t height, uint32_t width, DeviceMode deviceMode)
            : OperatorContext(inputTensorRefs, outputTensorRefs),
              top(top),
              left(left),
              height(height),
              width(width),
              deviceMode(deviceMode)
        {
        }
    };

    struct NormalizeContext : OperatorContext {
        std::vector<float> mean;
        std::vector<float> stddev;
        DeviceMode deviceMode;
        NormalizeContext(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                         const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs,
                         const std::vector<float> mean, const std::vector<float> stddev, DeviceMode deviceMode)
            : OperatorContext(inputTensorRefs, outputTensorRefs), mean(mean), stddev(stddev), deviceMode(deviceMode)
        {
        }
    };

    struct QwenFusionContext : OperatorContext {
        std::vector<float> mean;   // Mean values for normalization
        std::vector<float> std;    // Std values for normalization
        int resizeH;               // Target height for resize
        int resizeW;               // Target width for resize
        TensorFormat layout;       // Tensor layout format (e.g., NHWC)
        DeviceMode deviceMode;     // Device mode (CPU/NPU/GPU etc.)

        QwenFusionContext(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                          const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs,
                          const std::vector<float>& mean,
                          const std::vector<float>& std,
                          int resizeH,
                          int resizeW,
                          TensorFormat layout,
                          DeviceMode deviceMode)
            : OperatorContext(inputTensorRefs, outputTensorRefs),
              mean(mean),
              std(std),
              resizeH(resizeH),
              resizeW(resizeW),
              layout(layout),
              deviceMode(deviceMode)
        {
        }
    };

    struct ToTensorContext : OperatorContext {
        TensorFormat format; // target convert format
        DeviceMode deviceMode;
        ToTensorContext(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                        const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs,
                        TensorFormat format, DeviceMode deviceMode)
            : OperatorContext(inputTensorRefs, outputTensorRefs), format(format), deviceMode(deviceMode)
        {
        }
    };
}

#endif // OPERATOR_CONTEXT_H