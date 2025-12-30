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
 * @Date: 2025-2-11 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-11 17:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_OPERATOR_H_
#define ACCDATA_SRC_CPP_OPERATOR_OPERATOR_H_

#include "op_spec.h"
#include "pipeline/workspace/workspace.h"

namespace acclib {
namespace accdata {

class Operator {
public:
    Operator(const OpSpec& spec) : mSpec(spec)
    {
    }

    virtual ~Operator() = default;

    virtual AccDataErrorCode Run(Workspace& ws) = 0;

    const OpSpec& GetSpec()
    {
        return mSpec;
    }

private:
    OpSpec mSpec;
};

} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_OPERATOR_H_
