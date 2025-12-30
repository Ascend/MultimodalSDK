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
 * Description: Processing of the Tensor Class.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/tensor/Tensor.h"
#include <iostream>
#include <numeric>
#include <climits>
#include <algorithm>
#include <iostream>
#include "securec.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace {
constexpr int32_t CAN_ACCESS_FLAG = 1;
constexpr int32_t FOUR_DIM = 4;
constexpr int32_t DEVICE_CPU = -1;
} // namespace
namespace Acc {
void Tensor::FillAuxInfo()
{
    // Calculate caches
    auxInfo_.elementNums =
        std::accumulate(shape_.begin(), shape_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auxInfo_.perElementBytes = GetByteSize(dataType_);
    auxInfo_.totalBytes = auxInfo_.elementNums * auxInfo_.perElementBytes;
    uint32_t currentStrideBase = 1;
    auxInfo_.memoryStrides.resize(shape_.size());
    auxInfo_.logicalStrides.resize(shape_.size());
    for (size_t i = shape_.size(); i > 0; i--) {
        if (UINT_MAX / auxInfo_.perElementBytes < currentStrideBase) {
            LogError << "Get invalid tensor shape. The shape exceeds UINT_MAX." << GetErrorInfo(ERR_OUT_OF_RANGE);
            throw std::runtime_error("Get invalid tensor shape.");
        }
        auxInfo_.memoryStrides[i - 1] = auxInfo_.perElementBytes * currentStrideBase;
        auxInfo_.logicalStrides[i - 1] = currentStrideBase;
        if (shape_[i - 1] != 0 && UINT_MAX / shape_[i - 1] < currentStrideBase) {
            LogError << "Get invalid tensor shape. The shape exceeds UINT_MAX." << GetErrorInfo(ERR_OUT_OF_RANGE);
            throw std::runtime_error("Get invalid tensor shape.");
        }
        currentStrideBase *= shape_[i - 1];
    }
}

void Tensor::CheckTensorParams()
{
    if (dataPtr_ == nullptr) {
        LogError << "Illegal data. Data should not be nullptr." << GetErrorInfo(ERR_INVALID_POINTER);
        throw std::runtime_error("Invalid parameter.");
    }
    if (shape_.size() == 0) {
        LogError << "Illegal shape. Shape size should not be 0." << GetErrorInfo(ERR_INVALID_PARAM);
        throw std::runtime_error("Invalid parameter.");
    }
    if (format_ != TensorFormat::ND && shape_.size() != FOUR_DIM) {
        LogError << "Illegal shape. Shape size must be 4 when tensor format is NHWC or NCHW."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        throw std::runtime_error("Invalid parameter.");
    }
    if (device_ != "cpu") {
        LogError << "Illegal device. Only cpu are supported now." << GetErrorInfo(ERR_UNSUPPORTED_TYPE);
        throw std::runtime_error("Invalid parameter.");
    }
}

Tensor::Tensor(void* data, const std::vector<size_t>& shape, DataType dataType, TensorFormat format, const char* device)
    : deviceId_(DEVICE_CPU),
      shape_(shape),
      dataType_(dataType),
      format_(format),
      dataPtr_(std::shared_ptr<void>(data, [](void*) {})),
      device_(device ? device : "")
{
    CheckTensorParams();
    FillAuxInfo();
}

Tensor::Tensor(std::shared_ptr<void> dataPtr, const std::vector<size_t>& shape, DataType dataType, TensorFormat format,
               const char* device)
    : deviceId_(DEVICE_CPU),
      shape_(shape),
      dataType_(dataType),
      format_(format),
      dataPtr_(dataPtr),
      device_(device ? device : "")
{
    CheckTensorParams();
    FillAuxInfo();
}

ErrorCode Tensor::Clone(Tensor& tensor) const
{
    if (dataPtr_ == nullptr || auxInfo_.totalBytes == 0) {
        LogWarn << "Current tensor is empty, the clone operation is invalid.";
        return SUCCESS;
    }
    // Malloc dst memory and copy scr memory to dst memory
    char* data = new(std::nothrow) char[auxInfo_.totalBytes];
    if (data == nullptr) {
        LogError << "Failed to malloc for tensor." << GetErrorInfo(ERR_BAD_ALLOC);
        return ERR_BAD_ALLOC;
    }
    std::shared_ptr<void> dstPtr(static_cast<void*>(data), [](void* ptr) { delete[] static_cast<char*>(ptr); });
    auto ret = memcpy_s(dstPtr.get(), auxInfo_.totalBytes, dataPtr_.get(), auxInfo_.totalBytes);
    if (ret != SUCCESS) {
        LogError << "Tensor clone failed, may be caused by out of memory, please check the memory status of the "
                 << "environment." << GetErrorInfo(ERR_BAD_COPY);
        return ERR_BAD_COPY;
    }
    tensor = Tensor(dstPtr, shape_, dataType_, format_, this->Device().get());
    return SUCCESS;
}

ErrorCode Tensor::SetFormat(TensorFormat tensorFormat)
{
    if (tensorFormat == format_) {
        return SUCCESS;
    }
    if (tensorFormat == TensorFormat::ND) {
        format_ = tensorFormat;
        return SUCCESS;
    }
    if (shape_.size() == FOUR_DIM && (tensorFormat == TensorFormat::NHWC || tensorFormat == TensorFormat::NCHW)) {
        format_ = tensorFormat;
        return SUCCESS;
    }
    LogError << "Fail to set tensor format NHWC or NCHW, tensor shape size must be 4 when set format to NHWC or NCHW."
             << GetErrorInfo(ERR_INVALID_PARAM);
    return ERR_INVALID_PARAM;
}

const std::vector<size_t>& Tensor::Shape() const
{
    return shape_;
}

DataType Tensor::DType() const
{
    return dataType_;
}

std::unique_ptr<char[]> Tensor::Device() const
{
    auto ptr = std::make_unique<char[]>(device_.size() + 1);
    auto ret = strcpy_s(ptr.get(), device_.size() + 1, device_.c_str());
    if (ret != 0) {
        LogError << "Get device string failed, may be caused by out of memory, please check the memory status of the"
                 << "environment." << GetErrorInfo(ERR_BAD_COPY);
        return nullptr;
    }
    return ptr;
};

TensorFormat Tensor::Format() const
{
    return format_;
}

size_t Tensor::NumBytes() const
{
    return auxInfo_.totalBytes;
}

void* Tensor::Ptr() const
{
    return dataPtr_.get();
}

std::shared_ptr<void> Tensor::SharedPtr() const
{
    return dataPtr_;
}

TensorAuxInfo Tensor::AuxInfo() const
{
    return auxInfo_;
}
} // namespace Acc