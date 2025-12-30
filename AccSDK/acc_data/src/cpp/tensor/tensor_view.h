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

#ifndef ACCDATA_SRC_CPP_TENSOR_TENSOR_VIEW_H_
#define ACCDATA_SRC_CPP_TENSOR_TENSOR_VIEW_H_

#include <vector>

#include "tensor.h"

namespace acclib {
namespace accdata {

class TensorLayoutView {
public:
    /* * @brief View of tensors layouts. */
    explicit TensorLayoutView(const std::vector<Tensor> &tensors) : mCount(tensors.size()), mTensors(&tensors) {}

    /* * @brief View of one layouts. */
    TensorLayoutView(uint64_t count, TensorLayout layout) : mCount(count), mLayout(layout) {}

    uint64_t Size() const
    {
        return mCount;
    }

    TensorLayout operator[](uint64_t i) const
    {
        if (mTensors != nullptr) {
            return (*mTensors)[i].Layout();
        }
        return mLayout;
    }

private:
    uint64_t mCount{ 0 };
    const std::vector<Tensor> *mTensors = nullptr;
    TensorLayout mLayout{ TensorLayout::LAST };
};

class TensorDataTypeView {
public:
    /* * @brief View of tensors datatypes. */
    explicit TensorDataTypeView(const std::vector<Tensor> &tensors) : mCount(tensors.size()), mTensors(&tensors) {}

    /* * @brief View of one datatype. */
    TensorDataTypeView(uint64_t count, TensorDataType dataType) : mCount(count), mDataType(dataType) {}

    uint64_t Size() const
    {
        return mCount;
    }

    TensorDataType operator[](uint64_t i) const
    {
        if (mTensors != nullptr) {
            return (*mTensors)[i].DataType();
        }
        return mDataType;
    }

private:
    uint64_t mCount{ 0 };
    const std::vector<Tensor> *mTensors = nullptr;
    TensorDataType mDataType{ TensorDataType::LAST };
};

class TensorShapeView {
public:
    /* * @brief View of tensors shapes. */
    explicit TensorShapeView(const std::vector<Tensor> &tensors) : mCount(tensors.size()), mTensors(&tensors) {}

    /* * @brief View of one shape. */
    TensorShapeView(uint64_t count, const TensorShape &shape) : mCount(count), mSingleShape(&shape) {}

    uint64_t Size() const
    {
        return mCount;
    }

    const TensorShape &operator[](uint64_t i) const
    {
        if (mTensors != nullptr) {
            return (*mTensors)[i].Shape();
        }
        return *mSingleShape;
    }

private:
    uint64_t mCount{ 0 };
    const std::vector<Tensor> *mTensors = nullptr;
    const TensorShape *mSingleShape = nullptr;
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_TENSOR_TENSOR_VIEW_H_
