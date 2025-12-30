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
#ifndef ACCDATA_SRC_CPP_OPERATOR_MATH_NORMALIZE_ARGS_H_
#define ACCDATA_SRC_CPP_OPERATOR_MATH_NORMALIZE_ARGS_H_

#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"

namespace acclib {
namespace accdata {

/**
 * @brief Arguments for normalize operation.
 *
 * Arguments that can be specified through OpSpec:
 *  Required arguments:
 *      - mean: Mean value to be subtracted from the data.
 *      - stddev: Standard deviation value to scale the data.
 * Optional arguments:
 *      - scale: The scaling factor applied to the output. Default is 1.0.
 */
class NormalizeArgs {
public:
    NormalizeArgs() = default;
    ~NormalizeArgs() = default;

    /**
     * @brief Prepare arguments from OpSpec and Workspace.
     * @note Must be called before other member functions.
     */
    AccDataErrorCode Setup(const OpSpec &spec, Workspace &ws);

    /** @brief Mean value to be subtracted from the data. */
    const std::vector<float>& Mean()
    {
        return mMean;
    }

    /**
     * @brief Factors to scale the data.
     *
     * Factors = 1 / stddev * scale
     */
    const std::vector<float>& Scale()
    {
        return mStddev;
    }

private:
    std::vector<float> mMean{};
    std::vector<float> mStddev{};
};

} // namespace accdata
} // namespace acclib
#endif  // ACCDATA_SRC_CPP_OPERATOR_MATH_NORMALIZE_ARGS_H_
