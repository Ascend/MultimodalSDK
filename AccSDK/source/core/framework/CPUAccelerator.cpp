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
 * Description: CPUAccelerator API.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/CPUAccelerator.h"

namespace Acc {
CPUAccelerator::CPUAccelerator()
{
    operatorMap_[OperatorId::CROP] = CreateOperatorFunc<CropContext>(CPUAccelerator::Crop);
    operatorMap_[OperatorId::QWENFUSION] = CreateOperatorFunc<QwenFusionContext>(CPUAccelerator::QwenFusionOperator);
    operatorMap_[OperatorId::TOTENSOR] = CreateOperatorFunc<ToTensorContext>(CPUAccelerator::ToTensor);
    operatorMap_[OperatorId::NORMALIZE] = CreateOperatorFunc<NormalizeContext>(CPUAccelerator::Normalize);
    operatorMap_[OperatorId::RESIZE] = CreateOperatorFunc<ResizeContext>(CPUAccelerator::Resize);
}
} // namespace Acc