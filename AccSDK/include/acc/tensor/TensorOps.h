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
 * Description: Head file for processing of the Tensor Function.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <optional>
#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorDataType.h"

namespace Acc {

/**
 * @description: Tensor Crop.
 * @param src: Input tensor.
 * @param dst: Output tensor.
 * @param top: Top boundary position of the crop.
 * @param left: Left boundary position of the crop.
 * @param height: Crop height.
 * @param width: Crop width.
 * @param deviceMode: The mode for running operator.
 */
ErrorCode TensorCrop(const Tensor& src, Tensor& dst, uint32_t top, uint32_t left, uint32_t height, uint32_t width,
                     DeviceMode deviceMode = DeviceMode::CPU);

/**
 * @description: Tensor Resize.
 * @param src: Input tensor.
 * @param dst: Output tensor.
 * @param resizedH: resized height.
 * @param resizedW: resize width.
 * @param interpolation: interpolation algorithm.
 * @param deviceMode: The mode for running operator.
 */
ErrorCode TensorResize(const Tensor& src, Tensor& dst, size_t resizedH, size_t resizedW,
                       Interpolation interpolation = Interpolation::BICUBIC, DeviceMode deviceMode = DeviceMode::CPU);

/**
 * @brief Normalizes input tensor using mean and standard deviation values.
 *        Applies the formula: output = (input - mean) / std for each channel.
 * @param src Input tensor to be normalized.
 * @param dst Output tensor to store the normalized result.
 * @param mean Vector of mean values for normalization, one value per channel.
 * @param std Vector of standard deviation values for normalization, one value per channel.
 * @param deviceMode Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.
 * @return ErrorCode
 */
ErrorCode TensorNormalize(const Tensor& src, Tensor& dst, const std::vector<float>& mean, const std::vector<float>& std,
                          DeviceMode deviceMode = DeviceMode::CPU);

/**
 * @brief Converts a tensor from one format to another, equivalent to torchvision.transforms.ToTensor
 * @param src Input tensor to be 'to tensor'.
 * @param dst  Output tensor to store the result.
 * @param format The target tensor format specifying the desired layout and supports [NHWC/NCHW]
 * @param deviceMode Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.
 * @return ErrorCode
 */
ErrorCode TensorToTensor(const Tensor& src, Tensor& dst, TensorFormat format,
                         DeviceMode deviceMode = DeviceMode::CPU);
} // namespace Acc

#endif // TENSOR_OPS_H
