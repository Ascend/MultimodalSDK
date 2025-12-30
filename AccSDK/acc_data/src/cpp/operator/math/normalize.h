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
#ifndef ACCDATA_SRC_CPP_OPERATOR_MATH_NORMALIZE_H_
#define ACCDATA_SRC_CPP_OPERATOR_MATH_NORMALIZE_H_

#include "operator/operator.h"
#include "normalize_args.h"
#include "tensor/tensor_image.h"
#include "operator/operator_param_inner.h"

namespace acclib {
namespace accdata {

/**
 * @brief Normalize
 *
 * Normalizes the input by removing the mean and dividing by the standard deviation.
 *      Formula: out = (in - mean) / stddev * scale
 * SCHEMA BEGIN
 *      Inputs:
 *          - 0, original data.
 *      Outputs:
 *          - 0, Normalized data.
 *      Argument:
 *          - @see NormalizeArgs
 * SCHEMA END
 */
class Normalize : public Operator {
public:
    explicit Normalize(const OpSpec &spec)
        : Operator(spec)
    {
    }

    ~Normalize() = default;

    AccDataErrorCode Run(Workspace &ws) override;

private:
    AccDataErrorCode Setup(Workspace &ws);

    void SetupParam(OperatorParam &param);

    AccDataErrorCode AddTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template<typename InputType, typename OutputType>
    AccDataErrorCode AddTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template<typename InputType, typename OutputType, TensorLayout Layout>
    AccDataErrorCode AddTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template<typename InputType, typename OutputType, TensorLayout Layout>
    void RunTask(const InputType *input, const OperatorParam &param, OutputType *output, AccDataErrorCode &errCode);

    template<typename InputType, typename OutputType>
    void RunHWC(const InputType *input, const OperatorParam &param, OutputType *output);

    template<typename InputType, typename OutputType>
    void RunCHW(const InputType *input, const OperatorParam &param, OutputType *output);

private:
    image::Meta mInputMeta;
    NormalizeArgs mNormalizeArgs;
};

} // namespace accdata
} // namespace acclib
#endif  // ACCDATA_SRC_CPP_OPERATOR_MATH_NORMALIZE_H_
