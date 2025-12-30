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
 * Description: XPUAccelerator API.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/XPUAccelerator.h"
#include "acc/core/framework/CPUAccelerator.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"

namespace Acc {
XPUAccelerator& GetAccelerator(DeviceMode device)
{
    switch (device) {
        case DeviceMode::CPU:
            return CPUAccelerator::GetInstance();
        default:
            throw std::invalid_argument("Unknown device mode");
    }
}

ErrorCode XPUAccelerator::ExecuteOperator(OperatorId opId, OperatorContext& opCtx)
{
    auto it = operatorMap_.find(opId);
    if (it == operatorMap_.end()) {
        LogDebug << "The expected operator is not registry in "
                    "XPUAccelerator"<< GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    return it->second(opCtx);
}
} // namespace Acc