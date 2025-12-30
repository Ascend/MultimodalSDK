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
 * Description: Source file for operator custom check.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/tensor/OpsCustomChecker.h"
#include <numeric>
#include "acc/utils/LogImpl.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/utils/ErrorCodeUtils.h"
using namespace Acc;

namespace {
constexpr size_t MIN_WIDTH = 10;
constexpr size_t MIN_HEIGHT = 10;
constexpr size_t HEIGHT_INDEX_NCHW = 2;
constexpr size_t HEIGHT_INDEX_NHWC = 1;
constexpr size_t CHANNEL_INDEX_NHWC = 3;
constexpr size_t CHANNEL_INDEX_NCHW = 1;
constexpr size_t MAX_WIDTH = 8192;
constexpr size_t MAX_HEIGHT = 8192;
constexpr size_t MEAN_STD_SIZE = 3;
} // namespace

namespace Acc {
namespace {
ErrorCode CheckCropInputOutputConsistency(const CropContext& ctx)
{
    auto& src = ctx.inputTensorRefs[0].get();
    auto& outputTensor = ctx.outputTensorRefs[0].get();
    if (src.DType() != outputTensor.DType()) {
        LogError << "The datatype of input and output should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (src.Format() != outputTensor.Format()) {
        LogError << "The format of input and output should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (src.Shape()[0] != outputTensor.Shape()[0]) {
        LogError << "The batch size of input and output should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (src.Shape()[CHANNEL_INDEX_NHWC] != outputTensor.Shape()[CHANNEL_INDEX_NHWC]) {
        LogError << "The channel size of input and output should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (ctx.height != outputTensor.Shape()[HEIGHT_INDEX_NHWC]) {
        LogError << "Resized height must be equal to the dst height." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (ctx.width != outputTensor.Shape()[HEIGHT_INDEX_NHWC + 1]) {
        LogError << "Resized width must be equal to the dst width." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}
} // namespace
ErrorCode ResizeChecker::CheckCustomRules(const OperatorContext& ctx)
{
    const auto* resizeCtx = dynamic_cast<const ResizeContext*>(&ctx);
    if (resizeCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    if (resizeCtx->deviceMode != DeviceMode::CPU) {
        LogError << "Unsupported device mode, only support CPU mode." << GetErrorInfo(ERR_UNSUPPORTED_TYPE);
        return ERR_UNSUPPORTED_TYPE;
    }
    if (resizeCtx->interpolation != Interpolation::BICUBIC) {
        LogError << "Unsupported interpolation algorithm, only support BICUBIC." << GetErrorInfo(ERR_UNSUPPORTED_TYPE);
        return ERR_UNSUPPORTED_TYPE;
    }
    if (resizeCtx->resizedH > MAX_HEIGHT || resizeCtx->resizedH < MIN_HEIGHT || resizeCtx->resizedW > MAX_WIDTH ||
        resizeCtx->resizedW < MIN_WIDTH) {
        LogError << "Current resize width is " << resizeCtx->resizedW << ", height is " << resizeCtx->resizedH
                 << ", but should be range from [" << MIN_WIDTH << "," << MIN_HEIGHT << "] to [" << MAX_WIDTH << ","
                 << MAX_HEIGHT << "]." << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    if (!outputMallocFlags_[0]) {
        return SUCCESS;
    }
    auto& dst = resizeCtx->outputTensorRefs[0].get();
    auto& src = resizeCtx->inputTensorRefs[0].get();
    auto heightIndex = HEIGHT_INDEX_NHWC;
    if (dst.Shape()[heightIndex] != resizeCtx->resizedH) {
        LogError << "The height of dst should be equal to resizedH." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (dst.Shape()[heightIndex + 1] != resizeCtx->resizedW) {
        LogError << "The width of dst should be equal to resizedW." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (dst.DType() != src.DType()) {
        LogError << "The datatype of src and dst should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (dst.Format() != src.Format()) {
        LogError << "The format of src and dst should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode ResizeChecker::ImplicitMalloc(const OperatorContext& ctx)
{
    if (outputMallocFlags_[0]) {
        return SUCCESS;
    }
    const auto* resizeCtx = dynamic_cast<const ResizeContext*>(&ctx);
    if (resizeCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    auto src = resizeCtx->inputTensorRefs[0].get();
    auto heightIndex = HEIGHT_INDEX_NHWC;
    std::vector<size_t> dstShape = src.Shape();
    dstShape[heightIndex] = resizeCtx->resizedH;
    dstShape[heightIndex + 1] = resizeCtx->resizedW;
    auto totalBytes =
        std::accumulate(dstShape.begin(), dstShape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
        GetByteSize(src.DType());
    char* data = new(std::nothrow) char[totalBytes];
    if (data == nullptr) {
        LogError << "Failed to malloc for tensor." << GetErrorInfo(ERR_BAD_ALLOC);
        return ERR_BAD_ALLOC;
    }
    std::shared_ptr<void> dstPtr(static_cast<void*>(data), [](void* ptr) { delete[] static_cast<char*>(ptr); });
    resizeCtx->outputTensorRefs[0].get() = Tensor(dstPtr, dstShape, src.DType(), src.Format(), src.Device().get());
    return SUCCESS;
}

ErrorCode CropChecker::CheckCustomRules(const OperatorContext& ctx)
{
    const auto* cropCtx = dynamic_cast<const CropContext*>(&ctx);
    if (cropCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    if (cropCtx->deviceMode != DeviceMode::CPU) {
        LogError << "Unsupported device mode, only support CPU mode." << GetErrorInfo(ERR_UNSUPPORTED_TYPE);
        return ERR_UNSUPPORTED_TYPE;
    }
    auto& src = cropCtx->inputTensorRefs[0].get();
    auto heightIndex = HEIGHT_INDEX_NHWC;
    if (cropCtx->top + cropCtx->height > src.Shape()[heightIndex]) {
        LogError << "Top plus height exceeds the height of src. Current top is " << cropCtx->top << ", height is "
                 << cropCtx->height << ", and the height of src is " << src.Shape()[heightIndex] << "."
                 << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    if (cropCtx->left + cropCtx->width > src.Shape()[heightIndex + 1]) {
        LogError << "Left plus width exceeds the width of src. Current left is " << cropCtx->left << ", width is "
                 << cropCtx->width << ", and the width of src is " << src.Shape()[heightIndex + 1] << "."
                 << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    if (cropCtx->height < MIN_HEIGHT || cropCtx->width < MIN_WIDTH) {
        LogError << "Resized height should be larger than " << MIN_HEIGHT << "and width should be larger than"
                 << MIN_WIDTH << "." << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    if (!outputMallocFlags_[0]) {
        return SUCCESS;
    }
    if (CheckCropInputOutputConsistency(*cropCtx) != SUCCESS) {
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode CropChecker::ImplicitMalloc(const OperatorContext& ctx)
{
    if (outputMallocFlags_[0]) {
        return SUCCESS;
    }
    const auto* cropCtx = dynamic_cast<const CropContext*>(&ctx);
    if (cropCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    auto src = cropCtx->inputTensorRefs[0].get();
    auto heightIndex = HEIGHT_INDEX_NHWC;
    std::vector<size_t> dstShape = src.Shape();
    dstShape[heightIndex] = cropCtx->height;
    dstShape[heightIndex + 1] = cropCtx->width;
    auto totalBytes =
        std::accumulate(dstShape.begin(), dstShape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
        GetByteSize(src.DType());
    char* data = new(std::nothrow) char[totalBytes];
    if (data == nullptr) {
        LogError << "Failed to malloc for tensor." << GetErrorInfo(ERR_BAD_ALLOC);
        return ERR_BAD_ALLOC;
    }
    std::shared_ptr<void> dstPtr(static_cast<void*>(data), [](void* ptr) { delete[] static_cast<char*>(ptr); });
    cropCtx->outputTensorRefs[0].get() = Tensor(dstPtr, dstShape, src.DType(), src.Format(), src.Device().get());
    return SUCCESS;
}

ErrorCode NormalizeChecker::CheckCustomRules(const Acc::OperatorContext& ctx)
{
    const auto* normalizeCtx = dynamic_cast<const NormalizeContext*>(&ctx);
    if (normalizeCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }

    if (normalizeCtx->deviceMode != DeviceMode::CPU) {
        LogError << "Unsupported device mode, only support CPU mode." << GetErrorInfo(ERR_UNSUPPORTED_TYPE);
        return ERR_UNSUPPORTED_TYPE;
    }

    for (size_t i = 0; i < normalizeCtx->inputTensorRefs.size(); i++) {
        auto tensor = normalizeCtx->inputTensorRefs[i].get();
        size_t channelIndex = (tensor.Format() == TensorFormat::NCHW) ? CHANNEL_INDEX_NCHW : CHANNEL_INDEX_NHWC;
        if (tensor.Shape()[channelIndex] != normalizeCtx->mean.size()) {
            LogError << "Invalid mean vector size: expected 3 channels in input." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }

        if (tensor.Shape()[channelIndex] != normalizeCtx->stddev.size()) {
            LogError << "Invalid std vector size: expected 3 channels in input." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }

    return SUCCESS;
}

ErrorCode QwenFusionChecker::CheckCustomRules(const OperatorContext& ctx)
{
    const auto* qwenCtx = dynamic_cast<const QwenFusionContext*>(&ctx);
    if (qwenCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    size_t numInputs = ctx.inputTensorRefs.size();
    if (numInputs == 0 || ctx.outputTensorRefs.size() != numInputs) {
        LogError << "Input/Output size mismatch, please check." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    if (qwenCtx->mean.size() != MEAN_STD_SIZE) {
        LogError << "The input mean's size must be 3, please check." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (qwenCtx->std.size() != MEAN_STD_SIZE) {
        LogError << "The input std's size must be 3, please check." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (static_cast<size_t>(qwenCtx->resizeH) > MAX_HEIGHT || static_cast<size_t>(qwenCtx->resizeH) < MIN_HEIGHT ||
        static_cast<size_t>(qwenCtx->resizeW) > MAX_WIDTH || static_cast<size_t>(qwenCtx->resizeW) < MIN_WIDTH) {
        LogError << "Current resize width is " << qwenCtx->resizeW << ", height is " << qwenCtx->resizeH
                 << ", but should be range from [" << MIN_WIDTH << "," << MIN_HEIGHT << "] to [" << MAX_WIDTH << ","
                 << MAX_HEIGHT << "]." << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    for (size_t i = 0; i < qwenCtx->std.size(); ++i) {
        if (qwenCtx->std[i] <= 0.0f) {
            LogError << "Invalid input: std values must all be > 0. "
                     << "(index " << i << ", std=" << qwenCtx->std[i] << ")" << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    return SUCCESS;
}

ErrorCode QwenFusionChecker::ImplicitMalloc(const OperatorContext& ctx)
{
    if (outputMallocFlags_[0]) {
        return SUCCESS;
    }
    const auto* qwenCtx = dynamic_cast<const QwenFusionContext*>(&ctx);
    if (qwenCtx == nullptr) {
        LogDebug << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    auto src = qwenCtx->inputTensorRefs[0].get();
    auto heightIndex = HEIGHT_INDEX_NHWC;
    std::vector<size_t> dstShape = src.Shape();
    dstShape[heightIndex] = qwenCtx->resizeH;
    dstShape[heightIndex + 1] = qwenCtx->resizeW;
    auto totalBytes =
        std::accumulate(dstShape.begin(), dstShape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
        GetByteSize(src.DType());
    char* data = new(std::nothrow) char[totalBytes];
    if (data == nullptr) {
        LogError << "Failed to malloc for tensor." << GetErrorInfo(ERR_BAD_ALLOC);
        return ERR_BAD_ALLOC;
    }
    std::shared_ptr<void> dstPtr(static_cast<void*>(data), [](void* ptr) { delete[] static_cast<char*>(ptr); });
    qwenCtx->outputTensorRefs[0].get() = Tensor(dstPtr, dstShape, src.DType(), src.Format(), src.Device().get());
    return SUCCESS;
}

ErrorCode ToTensorChecker::CheckCustomRules(const Acc::OperatorContext& ctx)
{
    const auto* toTensorCtx = dynamic_cast<const ToTensorContext*>(&ctx);
    if (toTensorCtx == nullptr) {
        LogError << "The class of ctx is wrong, please check." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }

    if (toTensorCtx->format != TensorFormat::NCHW && toTensorCtx->format != TensorFormat::NHWC) {
        LogError << "The input format is invalid, it must be in [NHWC/NCHW]." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    return SUCCESS;
}
} // namespace Acc