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
 * Description: Crop op on cpu.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/CPUAccelerator.h"
#include "securec.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"

namespace {
constexpr size_t INDEX_ONE = 1;
constexpr size_t INDEX_TWO = 2;
constexpr size_t INDEX_THREE = 3;
constexpr size_t THREE_CHANNEL = 3;
} // namespace
namespace Acc {
ErrorCode CPUAccelerator::Crop(CropContext& opCtx)
{
    const Tensor& src = opCtx.inputTensorRefs[0].get();
    Tensor& dst = opCtx.outputTensorRefs[0].get();
    uint32_t top = opCtx.top;
    uint32_t left = opCtx.left;

    size_t srcRowStride = src.Shape()[INDEX_TWO] * THREE_CHANNEL;
    size_t dstHeight = dst.Shape()[INDEX_ONE];
    size_t dstWidth = dst.Shape()[INDEX_TWO];
    size_t dstRowStride = dstWidth * THREE_CHANNEL;

    uint8_t* srcPtr = static_cast<uint8_t*>(src.Ptr());
    uint8_t* srcPtrCopyStart = &srcPtr[top * srcRowStride + left * THREE_CHANNEL];
    uint8_t* dstPtr = static_cast<uint8_t*>(dst.Ptr());

    for (size_t h = 0; h < dstHeight; ++h) {
        if (memcpy_s(dstPtr + h * dstRowStride, dstRowStride, srcPtrCopyStart + h * srcRowStride, dstRowStride) !=
            EOK) {
            LogDebug << "memcpy_s in crop failed." << GetErrorInfo(ERR_BAD_COPY);
            return ERR_BAD_COPY;
        }
    }
    return SUCCESS;
}
} // namespace Acc