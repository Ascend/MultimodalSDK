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

#ifndef IMAGE_OPS_H
#define IMAGE_OPS_H

#include "acc/image/Image.h"
#include "acc/tensor/TensorDataType.h"
namespace Acc {
/**
 * @description: Image Crop.
 * @param src: Input image.
 * @param dst: Output image.
 * @param top: Top boundary position of the crop.
 * @param left: Left boundary position of the crop.
 * @param height: Crop height.
 * @param width: Crop width.
 * @param deviceMode: The mode for running operator.
 */
ErrorCode ImageCrop(const Image& src, Image& dst, uint32_t top, uint32_t left, uint32_t height, uint32_t width,
                    DeviceMode deviceMode = DeviceMode::CPU);

/**
 * @description: Image Resize.
 * @param src: Input image.
 * @param dst: Output image.
 * @param resizeW: resize width.
 * @param resizeH: resized height.
 * @param interpolation: interpolation algorithm.
 * @param deviceMode: The mode for running operator.
 */
ErrorCode ImageResize(const Image& src, Image& dst, size_t resizeW, size_t resizeH,
                      Interpolation interpolation = Interpolation::BICUBIC, DeviceMode deviceMode = DeviceMode::CPU);
} // namespace Acc

#endif // IMAGE_OPS_H