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

#ifndef ACCDATA_SRC_CPP_TENSOR_TENSOR_LIST_H_
#define ACCDATA_SRC_CPP_TENSOR_TENSOR_LIST_H_

#include "tensor.h"
#include "tensor_view.h"

namespace acclib {
namespace accdata {
const size_t IMAGE_TENSOR_SHAPE = 4;
class TensorListShape {
public:
    TensorListShape() = default;

    TensorListShape(uint64_t num, const TensorShape &shapes) : mNumShapes(num), mShapes(num, shapes) {}

    explicit TensorListShape(const std::vector<TensorShape> &shapes) : mNumShapes(shapes.size()), mShapes(shapes) {}

    ~TensorListShape() = default;

    uint64_t Size() const
    {
        return mNumShapes;
    }

    const TensorShape &operator[](uint64_t idx) const
    {
        if (mShapes.size() == 1) {
            return mShapes[0];
        }
        return mShapes[idx];
    }

private:
    uint64_t mNumShapes{ 0 };
    std::vector<TensorShape> mShapes;
};

/**
 * @brief A list of non-uniformly shared Tensor.
 *
 * Tensors managed by the container can have different shapes, data types, and layouts. It provides
 * some convenient functions, such as allowing you to modify the tensor layout in batches.
 */
class TensorList : public AccDataTensorList {
public:
    TensorList() = default;

    TensorList(const TensorList &) = delete;

    TensorList &operator = (const TensorList &) = delete;

    TensorList(TensorList &&other)
    {
        *this = std::move(other);
    }

    TensorList &operator = (TensorList &&other)
    {
        if (this == &other) {
            return *this;
        }
        mTensors = std::move(other.mTensors);
        return *this;
    }

    explicit TensorList(uint64_t batchSize) : mTensors(batchSize) {}

    ~TensorList() = default;

    /* * @brief Copy data into tensor. */
    AccDataErrorCode Copy(const TensorList &other)
    {
        return CopyOrShare<false>(other);
    }

    AccDataErrorCode Copy(std::shared_ptr<TensorList> other)
    {
        return CopyOrShare<false>(other);
    }

    /* * @brief Share data with tensor. */
    AccDataErrorCode ShareData(const TensorList &other)
    {
        return CopyOrShare<true>(other);
    }

    AccDataErrorCode ShareData(std::shared_ptr<TensorList> other)
    {
        return CopyOrShare<true>(other);
    }

    /* * @brief Resize tensors. */
    AccDataErrorCode Resize(const TensorShapeView &shape, const TensorDataTypeView &dataType)
    {
        return ResizeInner(shape, dataType);
    }

    /* * @brief Resize tensors and all tensors have same datatype. */
    AccDataErrorCode Resize(const TensorShapeView &shape, TensorDataType dataType)
    {
        return Resize(shape, TensorDataTypeView(shape.Size(), dataType));
    }

    /* * @brief Resize tensors and all tensors have same datatype. */
    template <typename T>
    AccDataErrorCode Resize(const TensorShapeView &shape)
    {
        return Resize(shape, TensorDataTypeEnum<T>());
    }

    /* * @brief Resize tensors. */
    AccDataErrorCode Resize(const TensorListShape &shape, const TensorDataTypeView &dataType)
    {
        return ResizeInner(shape, dataType);
    }

    /* * @brief Resize tensors and all tensors have same datatype. */
    AccDataErrorCode Resize(const TensorListShape &shape, TensorDataType dataType)
    {
        return Resize(shape, TensorDataTypeView(shape.Size(), dataType));
    }

    /* * @brief Resize tensors and all tensors have same datatype. */
    template <typename T>
    AccDataErrorCode Resize(const TensorListShape &shape)
    {
        return Resize(shape, TensorDataTypeEnum<T>());
    }

    /* * @brief Set all tensors to the specified layout. */
    void SetLayout(TensorLayout layout)
    {
        for (auto &tensor : mTensors) {
            tensor.SetLayout(layout);
        }
        return;
    }

    /* * @brief Set layout based on another tenor list. */
    AccDataErrorCode SetLayout(const TensorLayoutView &layout)
    {
        uint64_t numTensors = NumTensors();
        if (numTensors != layout.Size()) {
            ACCDATA_ERROR("Number of layouts is not equal to the number of tensors.");
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        for (uint64_t i = 0; i < numTensors; ++i) {
            mTensors[i].SetLayout(layout[i]);
        }
        return AccDataErrorCode::H_OK;
    }

    uint64_t NumTensors() const
    {
        return mTensors.size();
    }

    Tensor &operator[](uint64_t idx)
    {
        return mTensors[idx];
    }

    const Tensor &operator[](uint64_t idx) const
    {
        return mTensors[idx];
    }

    const std::vector<Tensor> &Tensors() const
    {
        return mTensors;
    }

    TensorShapeView Shape() const
    {
        return TensorShapeView(mTensors);
    }

    bool IsEmpty() const
    {
        return NumTensors() == 0;
    }

    bool IsValid() const
    {
        for (size_t i = 0; i < mTensors.size(); ++i) {
            if (!mTensors[i].IsValid()) {
                return false;
            }
            if (mTensors[i].Shape() != mTensors[0].Shape() ||
                mTensors[i].DataType() != mTensors[0].DataType() ||
                mTensors[i].Layout() != mTensors[0].Layout()) {
                ACCDATA_ERROR("TensorList Shape/DataType/Layout is not same.");
                return false;
            }
        }
        return true;
    }
private:
    template <bool IS_SHARED>
    AccDataErrorCode CopyOrShare(const TensorList &other)
    {
        uint64_t numTensors = other.mTensors.size();
        mTensors.resize(numTensors);
        for (uint64_t i = 0; i < numTensors; ++i) {
            if constexpr (IS_SHARED) {
                mTensors[i].ShareData(other.mTensors[i]);
            } else {
                auto errCode = mTensors[i].Copy(other.mTensors[i]);
                ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to copy.", errCode);
            }
            mTensors[i].SetLayout(other.mTensors[i].Layout());
        }
        return AccDataErrorCode::H_OK;
    }

    template <bool IS_SHARED>
    AccDataErrorCode CopyOrShare(std::shared_ptr<TensorList> other)
    {
        uint64_t numTensors = other->mTensors.size();
        mTensors.resize(numTensors);
        for (uint64_t i = 0; i < numTensors; ++i) {
            if constexpr (IS_SHARED) {
                mTensors[i].ShareData(other->mTensors[i]);
            } else {
                auto errCode = mTensors[i].Copy(other->mTensors[i]);
                ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to copy.", errCode);
            }
            mTensors[i].SetLayout(other->mTensors[i].Layout());
        }
        return AccDataErrorCode::H_OK;
    }

    template <typename T>
    AccDataErrorCode ResizeInner(const T &shape, const TensorDataTypeView &dataType)
    {
        uint64_t numTensors = shape.Size();
        if (dataType.Size() != numTensors) {
            ACCDATA_ERROR("Number of datatypes is not equal to the number of tensors.");
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        mTensors.resize(numTensors);
        for (uint64_t i = 0; i < numTensors; ++i) {
            mTensors[i].Resize(shape[i], dataType[i]);
        }
        return AccDataErrorCode::H_OK;
    }

    std::vector<Tensor> mTensors{};
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_TENSOR_TENSOR_LIST_H_
