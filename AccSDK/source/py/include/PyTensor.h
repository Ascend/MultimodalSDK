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
#ifndef PYTENSOR_H
#define PYTENSOR_H
#include <vector>
#include <memory>
#include "Python.h"

#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorDataType.h"
namespace PyAcc {

class Tensor {
public:
    /**
     * @brief Construct a new Tensor object
     *
     */
    Tensor();
    /**
     * @brief Destroy the Tensor object
     *
     */
    ~Tensor() = default;
    /**
     * @brief Construct a new Tensor object
     *
     * @param other input tensor
     */
    Tensor(const Tensor& other) = default;
    /**
     * @brief Assign operation
     *
     * @param other
     * @return Tensor& input tensor
     */
    Tensor& operator=(const Tensor& other) = default;
    /**
     * @brief Tensor deep copy to a new tensor, exposed as a Python interface
     *
     * @return Tensor dst tensor
     */
    Tensor clone() const;
    /**
     * @brief Set the tensor format property, exposed as a Python interface
     *
     * @param tensorFormat input format, range is ND, NHWC, NCHW
     */
    void set_format(Acc::TensorFormat tensorFormat);
    /**
     * @brief Support conversion from numpy ndarray to Tensor instance
     * @param pyObj python numpy ndarray
     * @return Tensor
     */
    static Tensor from_numpy(PyObject* pyObj);
    /**
     * @brief Support conversion Tensor instance to numpy ndarray
     * @return python numpy ndarray with __array_interface__ dict
     */
    PyObject* numpy();
    /**
     * @brief Swig Python funciton: Normalizes input tensor using mean and standard deviation values.
     *        Applies the formula: output = (input - mean) / std for each channel.
     * @param mean Vector of mean values for normalization, one value per channel.
     * @param std Vector of standard deviation values for normalization, one value per channel.
     * @param deviceMode Specifies the device mode for computation (CPU, NPU, DVPP, etc). Default is CPU.
     * @return Tensor
     */
    Tensor normalize(const std::vector<float>& mean, const std::vector<float>& std,
                     const Acc::DeviceMode deviceMode = Acc::DeviceMode::CPU);

    // inner aux func, will not expose Python interfaces.
public:
    /**
     * @brief Set the Acc Tensor object
     *
     * @param src input tensor
     */
    void SetTensor(const Acc::Tensor& src);
    /**
     * @brief Get the Acc Tensor Ptr object
     *
     * @return std::shared_ptr<Acc::Tensor>
     */
    std::shared_ptr<Acc::Tensor> GetTensorPtr() const;
    /**
     * @brief Get shape property
     *
     * @return std::vector<size_t>
     */
    std::vector<size_t> Shape() const;
    /**
     * @brief Get data type property
     *
     * @return Acc::DataType
     */
    Acc::DataType Dtype() const;
    /**
     * @brief Get device property
     *
     * @return const std::string& Returns a string to simplify swig wrap and conversion to a
     * Python built-in type.
     */
    const std::string& Device();
    /**
     * @brief Get format property
     *
     * @return Acc::TensorFormat
     */
    Acc::TensorFormat Format() const;
    /**
     * @brief Get num bytes property
     *
     * @return size_t
     */
    size_t NumBytes() const;

private:
    std::shared_ptr<Acc::Tensor> tensor_ = nullptr;
    std::string deviceStr_ = "cpu";
};

} // namespace PyAcc

#endif // PYTENSOR_H
