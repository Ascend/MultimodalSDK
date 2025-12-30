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
 * Description: CPU Operators.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef CPU_ACCELERATOR_H
#define CPU_ACCELERATOR_H

#include "XPUAccelerator.h"
#include "OperatorContext.h"

namespace Acc {
/**
 * @brief CPU accelerator implementation class
 * @details Concrete implementation of XPUAccelerator that executes operations
 *          using CPU computational resources. Optimized for multi-core
 *          CPU architectures and provides fallback implementations for all operators.
 */
class CPUAccelerator : public XPUAccelerator {
public:
    static CPUAccelerator& GetInstance()
    {
        static CPUAccelerator instance;
        return instance;
    }

    /**
     * @brief Construct a new CPU Accelerator object
     * @details Initializes the CPU accelerator instance and populates the operator map
     *          with CPU-specific implementations of all supported operations
     *          such as resize, crop, normalize, to_tensor, etc.
     */
    CPUAccelerator();

private:
    /**
     * @brief CPU-based image crop implementation
     * @details Extracts a rectangular region from the input image using CPU processing. No parameters checking.
     * @param ctx CropContext, reference OperatorContext.h
     * @return ErrorCode
     */
    static ErrorCode Crop(CropContext& opCtx);
    /**
     * @brief CPU-based tensor normalize implementation, reference accdata
     * @param opCtx NormalizeContext, reference OperatorContext.h
     * @return ErrorCode
     */
    static ErrorCode Normalize(NormalizeContext& opCtx);

    /**
     * @brief Core QwenFusion operator using QwenFusionContext
     *
     * Performs Resize + ToTensor + Normalize for multiple input Tensors
     * @param ctx QwenFusionContext, reference OperatorContext.h
     * @return ErrorCode
     */
    static ErrorCode QwenFusionOperator(QwenFusionContext& opCtx);
    /**
     * @brief CPU-based image ToTensor implementation
     * @details  Converts input data to tensor format, equivalent to torchvision.transforms.ToTensor
     * @param opCtx ToTensorContext, reference OpratorContext.h
     * @return ErrorCode
     */
    static ErrorCode ToTensor(ToTensorContext& opCtx);
    /**
     * @description: Resize op on cpu using bicubic interpolation.
     * @param opCtx ResizeContext, reference OpratorContext.h
     */
    static ErrorCode Resize(ResizeContext& opCtx);
};
} // namespace Acc

#endif // CPU_ACCELERATOR_H