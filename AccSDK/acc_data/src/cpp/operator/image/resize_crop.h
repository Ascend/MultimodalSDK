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
 * @Date: 2025-2-10 19:00:00
 * @LastEditTime: 2025-2-10 19:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_CROP_H_
#define ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_CROP_H_

#include "operator/operator.h"
#include "tensor/tensor_image.h"
#include "crop_args.h"
#include "resize_args.h"

namespace acclib {
namespace accdata {

/**
 * @brief Resize and Crop images.
 *
 * Resize the images with the specified size, then crop a square in the center.
 * SCHEMA BEGIN
 * Inputs:
 * - 0, Original images
 * Outputs:
 * - 0, Resized images
 * Argument:
 * - resize_w: The width of the resized image.
 * - resize_h: The height of the resized image.
 * - resize_shorter: The length of the shorter dimension of the resized image.
 * - resize_longer: The length of the longer dimension of the resized image.
 * SCHEMA END
 */
class ResizeCrop : public Operator {
public:
    explicit ResizeCrop(const OpSpec &spec) : Operator(spec) {}

    ~ResizeCrop() = default;

    AccDataErrorCode Run(Workspace &ws) override;

private:
    AccDataErrorCode Setup(Workspace &ws);

    AccDataErrorCode TorchResizeCrop(Workspace &ws);

    AccDataErrorCode TorchFloatImpl(Tensor &result, const Tensor &input, Workspace &ws);

private:
    image::Meta mInputMeta;
    ResizeArgs mResizeArgs;
    CropArgs mCropArgs;
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_CROP_H_
