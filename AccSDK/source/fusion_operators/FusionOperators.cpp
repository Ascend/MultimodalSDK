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
 * Description: Fusion Operator Api.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/fusion_operators/FusionOperators.h"

#include "acc/tensor/TensorOps.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorDataType.h"
#include "acc/core/framework/XPUAccelerator.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/core/framework/OperatorIndex.h"
#include "acc/tensor/OpsCustomChecker.h"
#include "acc/tensor/OpsBaseChecker.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace Acc {
ErrorCode FusionOperator::Qwen2VLImagePreprocess(const std::vector<std::shared_ptr<Image>>& images,
                                                 const QwenPreprocessConfig& config, std::vector<Tensor>& outputTensors)
{
    if (images.empty()) {
        LogError << "Input images should not be empty!" << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    outputTensors.resize(images.size());

    std::vector<std::reference_wrapper<const Tensor>> inputRefs;
    std::vector<std::reference_wrapper<Tensor>> outputRefs;
    inputRefs.reserve(images.size());
    outputRefs.reserve(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
        inputRefs.push_back(std::cref(images[i]->GetTensor()));
        outputRefs.push_back(std::ref(outputTensors[i]));
    }

    QwenFusionContext opCtx(inputRefs, outputRefs, config.mean, config.std, config.resizeH, config.resizeW,
                            TensorFormat::NHWC, DeviceMode::CPU);

    ErrorCode ret = QwenFusionChecker(OperatorId::QWENFUSION).CheckAndImplicitMalloc(opCtx);
    if (ret != SUCCESS) {
        return ret;
    }

    auto accelerator = Acc::GetAccelerator(opCtx.deviceMode);
    return accelerator.ExecuteOperator(OperatorId::QWENFUSION, opCtx);
}

} // namespace Acc