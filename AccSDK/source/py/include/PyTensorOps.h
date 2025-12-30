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
 * Description: tensor file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef PYTENSOROPS_H
#define PYTENSOROPS_H

#include "PyTensor.h"

namespace PyAcc {
    /**
     * @brief Swig Python funciton: Normalizes input tensor using mean and standard deviation values.
     *        Applies the formula: output = (input - mean) / std for each channel.
     * @param src Input tensor to be normalized.
     * @param dst Output tensor to store the normalized result.
     * @param mean Vector of mean values for normalization, one value per channel.
     * @param std Vector of standard deviation values for normalization, one value per channel.
     * @param deviceMode Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.
     * @return ErrorCode
     */
    void normalize(const Tensor& src, Tensor& dst, const std::vector<float>& mean, const std::vector<float>& std,
                   const Acc::DeviceMode deviceMode = Acc::DeviceMode::CPU);
}

#endif // PYTENSOROPS_H
