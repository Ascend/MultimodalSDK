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
#include "PyTensor.h"

#include <iostream>
#include "Python.h"

#include "PyUtil.h"
#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorOps.h"
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"

namespace PyAcc {
Tensor::Tensor() : tensor_(std::make_shared<Acc::Tensor>()) {}
Tensor Tensor::clone() const
{
    Tensor tensor;
    Acc::ErrorCode ret = tensor_->Clone(*tensor.GetTensorPtr());
    if (ret != Acc::SUCCESS) {
        Acc::LogError << "Tensor clone failed.";
        throw std::runtime_error("Tensor clone failed.");
    }
    return tensor;
}

void Tensor::set_format(Acc::TensorFormat tensorFormat)
{
    Acc::ErrorCode ret = tensor_->SetFormat(tensorFormat);
    if (ret != Acc::SUCCESS) {
        Acc::LogError << "Tensor set_format failed.";
        throw std::runtime_error("Tensor set_format failed.");
    }
}

std::vector<size_t> Tensor::Shape() const
{
    return tensor_->Shape();
}

Acc::DataType Tensor::Dtype() const
{
    return tensor_->DType();
}

const std::string& Tensor::Device()
{
    std::unique_ptr<char[]> device = tensor_->Device();
    if (!device) {
        Acc::LogError << "Get device string failed, may be caused by out of memory, please check the memory status of "
                         "the environment.";
        deviceStr_ = "";
    } else {
        std::string tmpStr(device.get());
        deviceStr_ = tmpStr;
    }
    return deviceStr_;
}

Acc::TensorFormat Tensor::Format() const
{
    return tensor_->Format();
}

size_t Tensor::NumBytes() const
{
    return tensor_->NumBytes();
}

void Tensor::SetTensor(const Acc::Tensor& src)
{
    (*tensor_) = src;
}

std::shared_ptr<Acc::Tensor> Tensor::GetTensorPtr() const
{
    return tensor_;
}

Tensor Tensor::from_numpy(PyObject* pyObj)
{
    NumpyData numpyData = GetNumpyData(pyObj);
    Acc::Tensor accTensor(numpyData.dataPtr, numpyData.shape, numpyData.dataType, Acc::TensorFormat::ND, "cpu");
    Tensor tensor;
    tensor.SetTensor(accTensor);
    return tensor;
}

PyObject* Tensor::numpy()
{
    NumpyData numpyData;
    numpyData.dataType = tensor_->DType();
    numpyData.dataPtr = tensor_->Ptr();
    numpyData.shape = tensor_->Shape();

    try {
        PyObject* numpyDataDict = ToNumpy(numpyData);
        return numpyDataDict;
    } catch (...) {
        throw std::runtime_error("Failed to get __array_interface__ for numpy ndarray. "
                                 "Maybe Python interpreter may be out of memory. ");
    }
}

Tensor Tensor::normalize(const std::vector<float>& mean, const std::vector<float>& std,
                         const Acc::DeviceMode deviceMode)
{
    Acc::Tensor outputAccTensor;
    Acc::ErrorCode ret = Acc::TensorNormalize(*tensor_, outputAccTensor, mean, std, deviceMode);
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error("Failed to execute normalize operator, please ensure your inputs are valid.");
    }
    Tensor tensor;
    tensor.SetTensor(outputAccTensor);
    return tensor;
}
} // namespace PyAcc