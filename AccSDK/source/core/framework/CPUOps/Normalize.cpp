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
 * Description: Normalize op on cpu.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/CPUAccelerator.h"
#include "acc/core/framework/Pipeline.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
#include "accdata_tensor.h"
#include "accdata_op_spec.h"

using namespace acclib::accdata;
namespace {
constexpr size_t NORMALIZE_THREAD_NUM = 1;
} // namespace
namespace Acc {
ErrorCode CPUAccelerator::Normalize(NormalizeContext& opCtx)
{
    Pipeline pipeline(NORMALIZE_THREAD_NUM);
    auto externalInput = AccDataOpSpec::Create("ExternalSource");
    if (!externalInput) {
        LogDebug << "Create ExternalSource specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }
    externalInput->AddOutput("ExternalSourceOutput", "cpu");
    auto normalize = AccDataOpSpec::Create("Normalize");
    if (!normalize) {
        LogDebug << "Create Normalize Operator specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }

    normalize->AddInput("ExternalSourceOutput", "cpu");
    normalize->AddArg("mean", opCtx.mean);
    normalize->AddArg("stddev", opCtx.stddev);
    normalize->AddOutput("NormalizeOutput", "cpu");

    ErrorCode ret = pipeline.Build({externalInput, normalize}, "NormalizeOutput");
    if (ret != SUCCESS) {
        return ret;
    }

    std::unordered_map<std::string, std::vector<Tensor>> inputs;
    auto& externalSourceOutput = inputs["ExternalSourceOutput"];
    externalSourceOutput.reserve(opCtx.inputTensorRefs.size());
    for (size_t i = 0; i < opCtx.inputTensorRefs.size(); i++) {
        externalSourceOutput.push_back(opCtx.inputTensorRefs[i].get());
    }

    Tensor& output = opCtx.outputTensorRefs[0].get();
    return pipeline.Run(inputs, output, false);
}
} // namespace Acc