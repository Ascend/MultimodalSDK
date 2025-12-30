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
#ifndef ACCDATA_SRC_CPP_OPERATOR_IMAGE_TO_TENSOR_H_
#define ACCDATA_SRC_CPP_OPERATOR_IMAGE_TO_TENSOR_H_

#include "common/utility.h"
#include "operator/operator_param_inner.h"
#include "operator/operator.h"
#include "tensor/tensor_image.h"
#include "to_tensor_args.h"

namespace acclib {
namespace accdata {

/**
 * @brief Transform images data type
 *
 * Transform images data type from uint8 to float32 by divide 255.0
 * SCHEMA BEGIN
 * Inputs:
 * - 0, Original images
 * Outputs:
 * - 0, Resized images
 * Argument:
 * - layout: the layout of the output data, which maybe 'NHWC' or 'NCHW'
 *
 * SCHEMA END
 */
class ToTensor : public Operator {
public:
    explicit ToTensor(const OpSpec &spec) : Operator(spec)
    {
    }

    ~ToTensor() = default;

    AccDataErrorCode Run(Workspace &ws) override;

private:
    AccDataErrorCode Setup(Workspace &ws);

    OperatorParam SetupParam();

    AccDataErrorCode GetOutputShape(const TensorList &input, TensorListShape &outputShape);

    template <typename InputType, typename OutputType>
    AccDataErrorCode AddTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template <typename InputType, typename OutputType, TensorLayout Layout>
    AccDataErrorCode AddTaskInner(ThreadPool &pool, const Tensor &input, Tensor &output);

    template <typename InputType, typename OutputType>
    AccDataErrorCode AddTaskSameLayout(ThreadPool &pool, const Tensor &input, Tensor &output);

    template <typename InputType, typename OutputType, TensorLayout Layout>
    void RunTask(const InputType *input, const OperatorParam param, OutputType *output, AccDataErrorCode &errCode);

    template <typename InputType, typename OutputType>
    void Trans2NCHW(const InputType *input, const OperatorParam &param, OutputType *output);

    template <typename InputType, typename OutputType>
    void Trans2NHWC(const InputType *input, const OperatorParam &param, OutputType *output);

private:
    ToTensorArgs mToTensorArgs;
    image::Meta mInputMeta;
};

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_IMAGE_TO_TENSOR_H_
