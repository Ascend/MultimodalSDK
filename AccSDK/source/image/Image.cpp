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
 * Description: Processing of the Image Class.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/image/Image.h"
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <iostream>
#include <memory>
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/tensor/Tensor.h"
#include "acc/utils/ImageUtils.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace {
using namespace Acc;
constexpr int32_t CPU_DEVICE_ID = -1;

constexpr size_t ONE_CHANNEL = 1;
constexpr size_t THREE_CHANNEL = 3;
constexpr size_t FOUR_CHANNEL = 4;
constexpr size_t FOUR_DIM = 4;
constexpr size_t ONE_BATCH = 1;

constexpr size_t INDEX_0 = 0;
constexpr size_t INDEX_1 = 1;
constexpr size_t INDEX_2 = 2;
constexpr size_t INDEX_3 = 3;

constexpr ErrorCode GetImageChannel(size_t& imChannel, ImageFormat imFormat)
{
    switch (imFormat) {
        case ImageFormat::RGB:
            [[fallthrough]];
        case ImageFormat::BGR:
            [[fallthrough]];
        case ImageFormat::RGB_PLANAR:
            [[fallthrough]];
        case ImageFormat::BGR_PLANAR:
            imChannel = THREE_CHANNEL;
            return SUCCESS;
        default:
            LogError << "Unsupported image format." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
    }
}

constexpr ErrorCode GetTensorFormatFromImage(TensorFormat& tensorFormat, ImageFormat imFormat)
{
    switch (imFormat) {
        case ImageFormat::RGB:
            [[fallthrough]];
        case ImageFormat::BGR:
            tensorFormat = TensorFormat::NHWC;
            return SUCCESS;
        case ImageFormat::RGB_PLANAR:
            [[fallthrough]];
        case ImageFormat::BGR_PLANAR:
            tensorFormat = TensorFormat::NCHW;
            return SUCCESS;
        default:
            LogError << "Unsupported image format." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
    }
}


ErrorCode GetTensorShapeFromImage(std::vector<size_t>& tensorShape, const std::vector<size_t>& imSize,
                                  ImageFormat imFormat)
{
    ErrorCode ret = CheckImSize(imSize);
    if (ret != SUCCESS) {
        LogError << "Check image size failed." << GetErrorInfo(ret);
        return ret;
    }

    // get image size
    size_t imBatch = 1;
    size_t imChannel;
    ret = GetImageChannel(imChannel, imFormat);
    if (ret != SUCCESS) {
        LogError << "Get image channel failed." << GetErrorInfo(ret);
        return ret;
    }
    size_t imWidth = imSize[0];
    size_t imHeight = imSize[1];

    // get tensor shape
    switch (imFormat) {
        case ImageFormat::RGB:
            [[fallthrough]];
        case ImageFormat::BGR:
            tensorShape = {imBatch, imHeight, imWidth, imChannel};
            return SUCCESS;
        case ImageFormat::RGB_PLANAR:
            [[fallthrough]];
        case ImageFormat::BGR_PLANAR:
            tensorShape = {imBatch, imChannel, imHeight, imWidth};
            return SUCCESS;
        default:
            LogError << "Unsupported image format." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
    }
}

void CheckDeviceFromConstructor(const char* device)
{
    if (device == nullptr || strcmp(device, "cpu") != 0) {
        LogError << "Illegal device. Only 'cpu' is supported now.";
        throw std::runtime_error("Invalid parameter: device must be 'cpu'.");
    }
}

void InitFromRawData(const std::vector<size_t>& imSize, ImageFormat imFormat, DataType dataType,
                     TensorFormat& tensorFormat, std::vector<size_t>& tensorShape)
{
    if (dataType != DataType::UINT8) {
        LogError << "Create image failed, Image datatype is invalid. Only support uint8.";
        throw std::runtime_error("Create image failed, Image datatype is invalid.");
    }

    ErrorCode ret = GetTensorFormatFromImage(tensorFormat, imFormat);
    if (ret != SUCCESS) {
        LogError << "Create image failed, Image format is invalid." << GetErrorInfo(ret);
        throw std::runtime_error("Create image failed, Image format is invalid.");
    }

    ret = GetTensorShapeFromImage(tensorShape, imSize, imFormat);
    if (ret != SUCCESS) {
        LogError << "Create image failed, image size or format is invalid." << GetErrorInfo(ret);
        throw std::runtime_error("Create image failed, image size or format is invalid.");
    }
}
} // namespace

namespace Acc {
Image::Image(void* data, const std::vector<size_t>& imSize, ImageFormat imFormat, DataType dataType, const char* device)
{
    CheckDeviceFromConstructor(device);
    TensorFormat tensorFormat = TensorFormat::ND;
    std::vector<size_t> tensorShape;
    InitFromRawData(imSize, imFormat, dataType, tensorFormat, tensorShape);
    tensor_ = Tensor(data, tensorShape, dataType, tensorFormat, device);
    size_ = imSize;
    format_ = imFormat;
}

Image::Image(std::shared_ptr<void> dataPtr, const std::vector<size_t>& imSize, ImageFormat imFormat, DataType dataType,
             const char* device)
{
    CheckDeviceFromConstructor(device);
    TensorFormat tensorFormat = TensorFormat::ND;
    std::vector<size_t> tensorShape;
    InitFromRawData(imSize, imFormat, dataType, tensorFormat, tensorShape);
    tensor_ = Tensor(dataPtr, tensorShape, dataType, tensorFormat, device);
    size_ = imSize;
    format_ = imFormat;
}

// Loaded through a path
Image::Image(const char* path, const char* device)
{
    LogDebug << "Create Image from path.";
    int imWidth;
    int imHeight;
    std::vector<uint8_t> imData;
    std::shared_ptr<unsigned char[]> ptr;
    CheckDeviceFromConstructor(device);
    auto decodeRet = ReadJpegData(path, imData, imWidth, imHeight, ptr);
    if (decodeRet != SUCCESS) {
        LogError << "ReadJpegData failed. Refer to the above log for detailed error information."
                 << GetErrorInfo(decodeRet);
        throw std::runtime_error("Create image from path failed. Failed to decode JPEG for CPU.");
    }
    size_ = {static_cast<size_t>(imWidth), static_cast<size_t>(imHeight)};
    format_ = ImageFormat::RGB;
    std::vector<size_t> dstShape = {ONE_BATCH, static_cast<size_t>(imHeight), static_cast<size_t>(imWidth),
                                    THREE_CHANNEL};
    std::shared_ptr<void> decodedRgbData(ptr.get(), [ptr](void*) mutable { ptr.reset(); });
    Tensor dst(decodedRgbData, dstShape, DataType::UINT8, TensorFormat::NHWC, "cpu");
    tensor_ = dst;
}

// Deep copy of data
ErrorCode Image::Clone(Image& other) const
{
    LogDebug << "Clone Image.";
    if (this == &other) {
        LogError << "Copy image failed. Clone source and target are the same.";
        return ERR_INVALID_PARAM;
    }

    ErrorCode ret = tensor_.Clone(other.tensor_);
    if (ret != SUCCESS) {
        LogError << "Copy image failed. See above for detailed error information." << GetErrorInfo(ret);
        return ret;
    }
    other.size_ = size_;
    other.format_ = format_;
    return SUCCESS;
}

// Obtain image attributes
const std::vector<size_t>& Image::Size() const
{
    return size_;
}

size_t Image::Width() const
{
    return size_[0];
}

size_t Image::Height() const
{
    return size_[1];
}

ImageFormat Image::Format() const
{
    return format_;
}

DataType Image::DType() const
{
    return tensor_.DType();
}

std::unique_ptr<char[]> Image::Device() const
{
    return tensor_.Device();
}

size_t Image::NumBytes() const
{
    return tensor_.NumBytes();
}

void* Image::Ptr() const
{
    return tensor_.Ptr();
}

Tensor& Image::GetTensor() const
{
    return tensor_;
}
} // namespace Acc