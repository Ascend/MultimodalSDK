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
#ifndef ACCDATA_SRC_CPP_OPERATOR_INPUT_EXTERNAL_SOURCE_H_
#define ACCDATA_SRC_CPP_OPERATOR_INPUT_EXTERNAL_SOURCE_H_

#include "operator/operator.h"
#include "common/cache_list.h"

namespace acclib {
namespace accdata {

/**
 * @brief External source
 *
 * It forwards external input to next OpNode. The external input is transferred through Feed().
 * SCHEMA BEGIN
 *      Inputs: None
 *      Outputs:
 *          - 0, External inputs
 * SCHEMA END
 */
class ExternalSource : public Operator {
public:
    explicit ExternalSource(const OpSpec &spec) : Operator(spec) {}

    ~ExternalSource() = default;

    AccDataErrorCode Run(Workspace &ws) override;

public:
    AccDataErrorCode Feed(std::shared_ptr<TensorList> input, bool copy);

private:
    CacheList<TensorList> mInputs{};
};

} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_INPUT_EXTERNAL_SOURCE_H_
