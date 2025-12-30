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

#ifndef ACCDATA_SRC_CPP_TENSOR_TENSOR_H_
#define ACCDATA_SRC_CPP_TENSOR_TENSOR_H_

#include <memory>
#include <vector>
#include <utility>
#include <numeric>
#include <unordered_set>

#include "securec.h"

#include "common/check.h"
#include "common/utility.h"
#include "accdata_tensor.h"

namespace acclib {
namespace accdata {

template <typename T>
inline constexpr TensorDataType TensorDataTypeEnum()
{
    if constexpr (std::is_same_v<T, uint8_t>) {
        return TensorDataType::UINT8;
    } else if constexpr (std::is_same_v<T, float>) {
        return TensorDataType::FP32;
    } else if constexpr (std::is_same_v<T, char>) {
        return TensorDataType::CHAR;
    } else {
        ACCDATA_ERROR("Unsupported data type.");
        return TensorDataType::LAST;
    }
}

inline int64_t NumElements(const TensorShape& shape, uint32_t dim = 0)
{
    if (dim >= shape.size()) {
        return 0;
    }
    
    return std::accumulate(shape.begin() + dim, shape.end(), 1ULL, std::multiplies<int64_t>());
}

inline bool IsValidShape(const TensorShape& shape)
{
    for (auto dim : shape) {
        if (dim <= 0) {
            return false;
        }
    }
    return true;
}

inline bool IsValidDataType(TensorDataType dataType)
{
    static const std::unordered_set<TensorDataType> validDataTypes = {
        TensorDataType::CHAR, TensorDataType::FP32, TensorDataType::UINT8
    };
    return validDataTypes.find(dataType) != validDataTypes.end();
}

inline bool IsValidLayout(TensorLayout layoutType)
{
    static const std::unordered_set<TensorLayout> validDataLayouts = {
        TensorLayout::NCHW, TensorLayout::NHWC, TensorLayout::PLAIN
    };
    return validDataLayouts.find(layoutType) != validDataLayouts.end();
}

/**
 * @brief Tensor
 */
class Tensor : public AccDataTensor {
public:
    Tensor() = default;

    Tensor(const Tensor &) = delete;

    Tensor &operator = (const Tensor &) = delete;

    Tensor(Tensor &&other)
    {
        *this = std::move(other);
    }

    Tensor &operator = (Tensor &&other)
    {
        if (this == &other) {
            return *this;
        }
        mLayout = std::exchange(other.mLayout, TensorLayout::PLAIN);
        mDataType = std::exchange(other.mDataType, TensorDataType::FP32);
        mShape = std::exchange(other.mShape, {});
        mNumBytes = std::exchange(mNumBytes, 0);
        mData = std::move(other.mData);
        return *this;
    }

    ~Tensor() = default;

    /* * @brief Copy data into tensor. */
    AccDataErrorCode Copy(const void *data, const TensorShape &shape, TensorDataType dataType)
    {
        if (data == nullptr || !IsValidShape(shape) || !IsValidDataType(dataType)) {
            ACCDATA_ERROR("Tensor copy failed: data is null, shape is valid, or data type is invalid.");
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        Resize(shape, dataType);
        const uint8_t *src = static_cast<const uint8_t *>(data);
        int64_t numBytes = NumElements(shape) * TensorDataTypeSize(dataType);
        auto ret = memcpy_s(RawDataPtr<uint8_t>(), mNumBytes, src, numBytes);
        if (ret != EOK) {
            ACCDATA_ERROR("Memcpy_s failed during copy.");
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        return AccDataErrorCode::H_OK;
    }

    /* * @brief Copy data into tensor. */
    template <typename T>
    AccDataErrorCode Copy(const T *data, const TensorShape &shape)
    {
        return Copy(data, shape, TensorDataTypeEnum<T>());
    }

    /* * @brief Copy data into tensor. */
    AccDataErrorCode Copy(const Tensor &other)
    {
        SetLayout(other.mLayout);
        return Copy(other.mData.get(), other.mShape, other.mDataType);
    }

    /* * @brief Share data with tensor. */
    AccDataErrorCode ShareData(const std::shared_ptr<void> &data, const TensorShape &shape, TensorDataType dataType)
    {
        if (data.get() == nullptr || !IsValidShape(shape) || !IsValidDataType(dataType)) {
            ACCDATA_ERROR("Tensor share failed: data is null, shape is valid, or data type is invalid.");
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        mDataType = dataType;
        mShape = shape;
        mNumBytes = NumElements(shape) * TensorDataTypeSize(dataType);
        mData = data;
        return AccDataErrorCode::H_OK;
    }

    /* * @brief Share data with tensor. */
    template <typename T>
    AccDataErrorCode ShareData(const std::shared_ptr<T> &data, const TensorShape &shape)
    {
        return ShareData(data, shape, TensorDataTypeEnum<T>());
    }

    /* * @brief Share data with tensor. */
    AccDataErrorCode ShareData(const Tensor &other)
    {
        SetLayout(other.Layout());
        return ShareData(other.mData, other.mShape, other.mDataType);
    }

    /* * @brief Resize the tensor. */
    void Resize(const TensorShape &shape, TensorDataType dataType)
    {
        mDataType = dataType;
        mShape = shape;
        int64_t numBytes = NumElements(shape) * TensorDataTypeSize(dataType);
        if (numBytes <= mNumBytes) {
            return;
        }
        mNumBytes = numBytes;
        mData = std::shared_ptr<uint8_t>(new (std::align_val_t(ACCDATA_ALIGN_SIZE)) uint8_t[mNumBytes],
            [](uint8_t* ptr) {  // 智能指针默认删除器处理自定义对齐分配的空间时可能导致内存泄漏
                operator delete[] (ptr, (std::align_val_t(ACCDATA_ALIGN_SIZE)));
            }
        );
        return;
    }

    /* * @brief Resize the tensor. */
    template <typename T>
    void Resize(const TensorShape &shape)
    {
        return Resize(shape, TensorDataTypeEnum<T>());
    }

    template <typename T>
    AccDataErrorCode Data(std::shared_ptr<T> &data) const
    {
        if (!IsDataType<T>()) {
            ACCDATA_ERROR("Different datatype.");
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        data = std::static_pointer_cast<T>(mData);
        return AccDataErrorCode::H_OK;
    }

    template <typename T> T *RawDataPtr() const
    {
        return static_cast<T *>(mData.get());
    }

    int64_t GetSize() const
    {
        return mNumBytes;
    }

    std::shared_ptr<void> RawDataPtr() const
    {
        return mData;
    }

    void Reset()
    {
        mLayout = TensorLayout::PLAIN;
        mDataType = TensorDataType::FP32;
        mShape.clear();
        mNumBytes = 0;
        mData.reset();
        return;
    }

    void SetLayout(TensorLayout layout)
    {
        mLayout = layout;
        return;
    }

    TensorLayout Layout() const
    {
        return mLayout;
    }

    TensorDataType DataType() const
    {
        return mDataType;
    }

    const TensorShape &Shape() const
    {
        return mShape;
    }

    template <typename T> bool IsDataType() const
    {
        return mDataType == TensorDataTypeEnum<T>();
    }

    bool IsValid() const
    {
        bool valid = (mData != nullptr && mNumBytes > 0 &&
               IsValidDataType(mDataType) && IsValidLayout(mLayout) &&
               IsValidShape(mShape) && mShape.size() == 4ULL); // 4 is NCHW/NHWC
        if (!valid) {
            ACCDATA_ERROR("Tensor is invalid, Info : [ IsValidDataType: " << IsValidDataType(mDataType) <<
            ", IsValidLayout: " << IsValidLayout(mLayout) <<
            ", IsValidShape: " << IsValidShape(mShape) && mShape.size() == 4ULL); // 4 is NCHW/NHWC
        }
        return valid;
    }

private:
    /* Informations to interpret the underlying storage */
    TensorLayout mLayout{ TensorLayout::LAST };
    TensorDataType mDataType{ TensorDataType::LAST };
    TensorShape mShape;

    /* Underlying storage */
    int64_t mNumBytes{ 0 };
    std::shared_ptr<void> mData{ nullptr };
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_TENSOR_TENSOR_H_
