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
 * Description: Tensor.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <stdexcept>
#include <vector>
#include "acc/ErrorCode.h"
#include "acc/tensor/TensorDataType.h"

constexpr size_t ONE_BYTE = 1;
constexpr size_t FOUR_BYTE = 4;

namespace Acc {
/**
 * @brief Get the Byte Size object
 *
 * @param type data type, range is FLOAT32, INT8, UINT8
 * @return constexpr size_t Number of bytes corresponding to the type
 */
constexpr size_t GetByteSize(DataType type)
{
    if (type == DataType::INT8 || type == DataType::UINT8) {
        return ONE_BYTE;
    } else if (type == DataType::FLOAT32) {
        return FOUR_BYTE;
    }
    throw std::runtime_error("Unsupported DataType.");
}

class Tensor {
public:
    /**
     * @brief Construct a new Tensor object
     *
     */
    Tensor() = default;
    /**
     * @brief Destroy the Tensor object
     *
     */
    ~Tensor() = default;
    /**
     * @brief Construct a new Tensor object
     *
     * @param other Input tensor
     */
    Tensor(const Tensor& other) = default;
    /**
     * @brief Assignment operation
     *
     * @param other Input tensor
     * @return Tensor& self tensor
     */
    Tensor& operator=(const Tensor& other) = default;
    /**
     * @brief Construct a new Tensor object
     *
     * @param dataPtr user input data
     * @param shape tensor shape
     * @param dataType data type, range is FLOAT32, INT8, UINT8
     * @param format layout format, range is ND, NHWC, NCHW
     * @param device device str, range is cpu
     */
    Tensor(std::shared_ptr<void> dataPtr, const std::vector<size_t>& shape, DataType dataType = DataType::FLOAT32,
           TensorFormat format = TensorFormat::ND, const char* device = "cpu");
    /**
     * @brief Construct a new Tensor object
     *
     * @param data user input data
     * @param shape tensor shape
     * @param dataType data type, range is FLOAT32, INT8, UINT8
     * @param format layout format, range is ND, NHWC, NCHW
     * @param device device str, range is cpu
     */
    Tensor(void* data, const std::vector<size_t>& shape, DataType dataType = DataType::FLOAT32,
           TensorFormat format = TensorFormat::ND, const char* device = "cpu");
    /**
     * @brief Tensor deep copy to a new tensor
     *
     * @param tensor dst tensor
     * @return ErrorCode
     */
    ErrorCode Clone(Tensor& tensor) const;
    /**
     * @brief Set the format
     *
     * @param tensorFormat input format, range is ND, NHWC, NCHW
     * @return ErrorCode
     */
    ErrorCode SetFormat(TensorFormat tensorFormat);
    /**
     * @brief Get shape property
     *
     * @return const std::vector<size_t>&
     */
    const std::vector<size_t>& Shape() const;
    /**
     * @brief Get data type property
     *
     * @return DataType
     */
    DataType DType() const;
    /**
     * @brief Get device property
     *
     * @return std::unique_ptr<char[]>
     */
    std::unique_ptr<char[]> Device() const;
    /**
     * @brief Get format property
     *
     * @return TensorFormat
     */
    TensorFormat Format() const;
    /**
     * @brief Get num bytes property
     *
     * @return size_t
     */
    size_t NumBytes() const;
    /**
     * @brief Get data ptr property
     *
     * @return void*
     */
    void* Ptr() const;
    /**
     * @brief Get shared data ptr property
     *
     * @return std::shared_ptr<void>
     */
    std::shared_ptr<void> SharedPtr() const;
    /**
     * @brief Get auxinfo property
     *
     * @return TensorAuxInfo
     */
    TensorAuxInfo AuxInfo() const;

private:
    /**
     * @brief Fill in auxiliary information in tensor constructor
     *
     */
    void FillAuxInfo();
    /**
     * @brief check params in tensor constructor
     *
     */
    void CheckTensorParams();

private:
    int32_t deviceId_ = -1;
    std::vector<size_t> shape_ = {};
    DataType dataType_ = DataType::FLOAT32;
    TensorFormat format_ = TensorFormat::ND;
    std::shared_ptr<void> dataPtr_ = nullptr;
    std::string device_ = "cpu";
    TensorAuxInfo auxInfo_ = {};
};
} // namespace Acc
#endif // TENSOR_H