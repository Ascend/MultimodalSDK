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

#include "PyTensorOps.h"

#include "acc/tensor/TensorOps.h"

namespace PyAcc {
    void normalize(const Tensor& src, Tensor& dst, const std::vector<float>& mean, const std::vector<float>& std,
                   const Acc::DeviceMode deviceMode)
    {
        std::shared_ptr<Acc::Tensor> srcAccTensor = src.GetTensorPtr();
        Acc::Tensor outputAccTensor;
        Acc::ErrorCode ret = Acc::TensorNormalize(*srcAccTensor.get(), outputAccTensor, mean, std, deviceMode);
        if (ret != Acc::SUCCESS) {
            throw std::runtime_error("Failed to execute normalize operator, please ensure your inputs are valid.");
        }
        dst.SetTensor(outputAccTensor);
    }
}