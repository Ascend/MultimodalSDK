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
 * Description: Processing of the Tensor Function.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/tensor/TensorOps.h"
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/core/framework/CPUAccelerator.h"
#include "acc/tensor/OpsCustomChecker.h"
#include "acc/tensor/OpsBaseChecker.h"
#include "acc/utils/ErrorCodeUtils.h"
using namespace Acc;

namespace Acc {
ErrorCode TensorCrop(const Tensor& src, Tensor& dst, uint32_t top, uint32_t left, uint32_t height, uint32_t width,
                     DeviceMode deviceMode)
{
    CropContext opCtx{{std::cref(src)}, {std::ref(dst)}, top, left, height, width, deviceMode};
    ErrorCode ret = CropChecker(OperatorId::CROP).CheckAndImplicitMalloc(opCtx);
    if (ret != SUCCESS) {
        return ret;
    }

    auto accelerator = Acc::GetAccelerator(deviceMode);
    return accelerator.ExecuteOperator(OperatorId::CROP, opCtx);
}

ErrorCode TensorResize(const Tensor& src, Tensor& dst, size_t resizedH, size_t resizedW, Interpolation interpolation,
                       DeviceMode deviceMode)
{
    ResizeContext opCtx{{std::cref(src)}, {std::ref(dst)}, resizedH, resizedW, interpolation, deviceMode};
    ErrorCode ret = ResizeChecker(OperatorId::RESIZE).CheckAndImplicitMalloc(opCtx);
    if (ret != SUCCESS) {
        return ret;
    }
    auto accelerator = Acc::GetAccelerator(deviceMode);
    return accelerator.ExecuteOperator(OperatorId::RESIZE, opCtx);
}

ErrorCode TensorNormalize(const Tensor& src, Tensor& dst, const std::vector<float>& mean, const std::vector<float>& std,
                          DeviceMode deviceMode)
{
    NormalizeContext opCtx{{std::cref(src)}, {std::ref(dst)}, mean, std, deviceMode};

    ErrorCode ret = NormalizeChecker(OperatorId::NORMALIZE).CheckAndImplicitMalloc(opCtx);
    if (ret != SUCCESS) {
        return ret;
    }

    auto accelerator = Acc::GetAccelerator(deviceMode);
    return accelerator.ExecuteOperator(OperatorId::NORMALIZE, opCtx);
}

ErrorCode TensorToTensor(const Tensor& src, Tensor& dst, TensorFormat format, DeviceMode deviceMode)
{
    ToTensorContext opCtx{{std::cref(src)}, {std::ref(dst)}, format, deviceMode};

    ErrorCode ret = ToTensorChecker(OperatorId::TOTENSOR).CheckAndImplicitMalloc(opCtx);
    if (ret != SUCCESS) {
        return ret;
    }
    auto accelerator = Acc::GetAccelerator(deviceMode);
    return accelerator.ExecuteOperator(OperatorId::TOTENSOR, opCtx);
}
} // namespace Acc