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
 * Description: Head file for operator custom check.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef OPS_CUSTOM_CHECK_H
#define OPS_CUSTOM_CHECK_H

#include "acc/tensor/TensorOps.h"
#include "acc/tensor/OpsBaseChecker.h"

namespace Acc {
class ResizeChecker : public OpsBaseChecker {
public:
    explicit ResizeChecker(const OperatorId& opId) : OpsBaseChecker(opId) {}

protected:
    ErrorCode CheckCustomRules(const OperatorContext& ctx) override;
    ErrorCode ImplicitMalloc(const OperatorContext& ctx) override;
};

class CropChecker : public OpsBaseChecker {
public:
    explicit CropChecker(const OperatorId& opId) : OpsBaseChecker(opId) {}

protected:
    ErrorCode CheckCustomRules(const OperatorContext& ctx) override;
    ErrorCode ImplicitMalloc(const OperatorContext& ctx) override;
};

class NormalizeChecker : public OpsBaseChecker {
public:
    explicit NormalizeChecker(const OperatorId& opId) : OpsBaseChecker(opId) {}

protected:
    ErrorCode CheckCustomRules(const OperatorContext& ctx) override;
};

class ToTensorChecker : public OpsBaseChecker {
public:
    explicit ToTensorChecker(const OperatorId& opId) : OpsBaseChecker(opId) {}

protected:
    ErrorCode CheckCustomRules(const OperatorContext& ctx) override;
};

class QwenFusionChecker : public OpsBaseChecker {
public:
    explicit QwenFusionChecker(const OperatorId& opId) : OpsBaseChecker(opId) {}

protected:
    ErrorCode CheckCustomRules(const OperatorContext& ctx) override;
    ErrorCode ImplicitMalloc(const OperatorContext& ctx) override;
};
} // namespace Acc
#endif // OPS_CUSTOM_CHECK_H
