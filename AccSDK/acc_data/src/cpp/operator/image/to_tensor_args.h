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
 * @Date: 2025-2-14 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-14 9:00:00
 */
#ifndef ACCDATA_SRC_CPP_OPERATOR_IMAGE_TO_TENSOR_ARGS_H_
#define ACCDATA_SRC_CPP_OPERATOR_IMAGE_TO_TENSOR_ARGS_H_

#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"
#include "tensor/tensor_image.h"

namespace acclib {
namespace accdata {

/**
 * @brief Arguments for to_tensor operation.
 *
 * Arguments that can be specified through OpSpec:
 * Optional argument:
 * - layout: The layout of the output which should be 'TensorLayout::NHWC' or 'TensorLayout::NCHW'.
 *          Default is 'TensorLayout::NHWC'.
 */
class ToTensorArgs {
public:
    static constexpr double NORM_FACTOR = 0.0039215686274509803921568627451;  // == 1/255.0

    ToTensorArgs() = default;

    ~ToTensorArgs() = default;

    /**
     * @brief   Prepare arguments from OpSpec and Workspace.
     * @note    Must be called before other member functions.
     */
    AccDataErrorCode Setup(const OpSpec &spec, Workspace &ws);

    /** @brief  layout of the output */
    inline TensorLayout Layout() const
    {
        return mLayout;
    }

    /** @brief  boolean value denotes whether the layout of output is same to input or not */
    inline bool IsSameLayout() const
    {
        return mSameLayout;
    }

    /** @brief  the factor to multiple to change an uint8_t to float, use 1/255.0 as default */
    inline double Mul() const
    {
        return NORM_FACTOR;
    }

private:
    TensorLayout mLayout{ TensorLayout::NCHW };
    bool mSameLayout{ true };
};

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_IMAGE_TO_TENSOR_ARGS_H_
