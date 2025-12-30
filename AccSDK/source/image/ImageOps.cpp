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
 * Description: Processing of the Image Function.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/image/ImageOps.h"
#include <iostream>
#include "acc/tensor/TensorOps.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace Acc {
namespace {
std::string ImageFormatToString(ImageFormat fmt)
{
    switch (fmt) {
        case ImageFormat::UNDEFINED:
            return "UNDEFINED";
        case ImageFormat::RGB:
            return "RGB";
        case ImageFormat::BGR:
            return "BGR";
        case ImageFormat::RGB_PLANAR:
            return "RGB_PLANAR";
        case ImageFormat::BGR_PLANAR:
            return "BGR_PLANAR";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(fmt)) + ")";
    }
}
} // namespace

ErrorCode ImageResize(const Image& src, Image& dst, size_t resizeW, size_t resizeH, Interpolation interpolation,
                      DeviceMode deviceMode)
{
    if (src.Format() != ImageFormat::RGB && src.Format() != ImageFormat::BGR) {
        LogError << "Current format is " << ImageFormatToString(src.Format()) << ", but should be RGB or BGR."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    auto ret = TensorResize(src.GetTensor(), dst.GetTensor(), resizeH, resizeW, interpolation, deviceMode);
    if (ret != SUCCESS) {
        LogError << "Image Resize failed. Please check the detailed log above for the cause." << GetErrorInfo(ret);
    } else {
        dst = Image(dst.GetTensor().SharedPtr(), {resizeW, resizeH}, src.Format(), DataType::UINT8);
    }
    return ret;
}

ErrorCode ImageCrop(const Image& src, Image& dst, uint32_t top, uint32_t left, uint32_t height, uint32_t width,
                    DeviceMode deviceMode)
{
    if (src.Format() != ImageFormat::RGB && src.Format() != ImageFormat::BGR) {
        LogError << "Current format is " << ImageFormatToString(src.Format()) << ", but should be RGB or BGR."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    auto ret = TensorCrop(src.GetTensor(), dst.GetTensor(), top, left, height, width, deviceMode);
    if (ret != SUCCESS) {
        LogError << "Image Crop failed. Please check the detailed log above for the cause." << GetErrorInfo(ret);
    } else {
        dst = Image(dst.GetTensor().SharedPtr(), {width, height}, src.Format(), DataType::UINT8);
    }
    return ret;
}
} // namespace Acc