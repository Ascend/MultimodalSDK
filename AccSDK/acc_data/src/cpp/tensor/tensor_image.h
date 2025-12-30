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
 * @Description:
 * @Version: 1.0
 * @Date: 2025-2-10 15:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-10 15:00:00
 */

#ifndef ACCDATA_SRC_CPP_TENSOR_TENSOR_IMAGE_H_
#define ACCDATA_SRC_CPP_TENSOR_TENSOR_IMAGE_H_

#include "tensor.h"

namespace acclib {
namespace accdata {

constexpr uint64_t MIN_PIXEL_LENGTH = 10;
constexpr uint64_t MAX_PIXEL_LENGTH = 8192;
namespace image {

inline AccDataErrorCode GetWidth(const Tensor &tensor, uint64_t &width)
{
    auto layout = tensor.Layout();
    auto shape = tensor.Shape();
    AccDataErrorCode err = AccDataErrorCode::H_OK;
    if (layout == TensorLayout::NHWC) {
        width = shape[2]; // 2-width dim
    } else if (layout == TensorLayout::NCHW) {
        width = shape[3]; // 3-width dim
    } else {
        ACCDATA_ERROR("Unsupported layout '" << layout << "'.");
        return AccDataErrorCode::H_TENSOR_ERROR;
    }
    if (width < MIN_PIXEL_LENGTH || width > MAX_PIXEL_LENGTH) {
        ACCDATA_ERROR("width should be in [" << MIN_PIXEL_LENGTH << ", "
            << MAX_PIXEL_LENGTH << "], but got " << width << ".");
        return AccDataErrorCode::H_TENSOR_ERROR;
    }
    return err;
}

inline AccDataErrorCode GetHeight(const Tensor &tensor, uint64_t &height)
{
    auto layout = tensor.Layout();
    auto shape = tensor.Shape();
    AccDataErrorCode err = AccDataErrorCode::H_OK;
    if (layout == TensorLayout::NHWC) {
        height = shape[1]; // 1-height dim
    } else if (layout == TensorLayout::NCHW) {
        height = shape[2]; // 2-height dim
    } else {
        ACCDATA_ERROR("Unsupported layout '" << layout << "'.");
        return AccDataErrorCode::H_TENSOR_ERROR;
    }
    if (height < MIN_PIXEL_LENGTH || height > MAX_PIXEL_LENGTH) {
        ACCDATA_ERROR("Height should be in [" << MIN_PIXEL_LENGTH << ", "
            << MAX_PIXEL_LENGTH << "], but got " << height << ".");
        err = AccDataErrorCode::H_TENSOR_ERROR;
    }
    return err;
}

inline AccDataErrorCode GetChannel(const Tensor &tensor, uint64_t &channel)
{
    auto layout = tensor.Layout();
    auto shape = tensor.Shape();
    AccDataErrorCode err = AccDataErrorCode::H_OK;
    if (layout == TensorLayout::NHWC) {
        channel = shape[3]; // 3-channel dim
    } else if (layout == TensorLayout::NCHW) {
        channel = shape[1]; // 1-channel dim
    } else {
        ACCDATA_ERROR("Unsupported layout '" << layout << "'.");
        err = AccDataErrorCode::H_TENSOR_ERROR;
    }
    if (channel != RGB_CHANNELS) {
        ACCDATA_ERROR("channel only support RGB_CHANNELS: 3! ");
        return AccDataErrorCode::H_TENSOR_ERROR;
    }
    return err;
}

class Meta {
public:
    Meta() = default;

    explicit Meta(const Tensor &tensor)
    {
        Setup(tensor);
    }

    ~Meta() = default;

    AccDataErrorCode Setup(const Tensor &tensor)
    {
        mNumSamples = tensor.Shape()[0];
        AccDataErrorCode errCode = image::GetHeight(tensor, mHeight);
        if (errCode != AccDataErrorCode::H_OK) {
            return errCode;
        }
        errCode = image::GetWidth(tensor, mWidth);
        if (errCode != AccDataErrorCode::H_OK) {
            return errCode;
        }
        errCode = image::GetChannel(tensor, mNumChannels);
        return errCode;
    }

    inline int64_t NumSamples() const
    {
        return mNumSamples;
    }

    inline int64_t Height() const
    {
        return mHeight;
    }

    inline int64_t Width() const
    {
        return mWidth;
    }

    inline int64_t NumChannels() const
    {
        return mNumChannels;
    }

private:
    uint64_t mNumSamples{ 0 };
    uint64_t mHeight{ 0 };
    uint64_t mWidth{ 0 };
    uint64_t mNumChannels{ 3 };
};

} // namespace image
} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_TENSOR_TENSOR_IMAGE_H_
