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
 * Description: ToTensor op on cpu.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/CPUAccelerator.h"
#include "acc/utils/TensorUtils.h"
#include "acc/core/framework/Pipeline.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
#include "accdata_tensor.h"
#include "accdata_op_spec.h"

using namespace acclib::accdata;
namespace {
constexpr size_t TO_TENSOR_THREAD_NUM = 1;
} // namespace
namespace Acc {
ErrorCode CPUAccelerator::ToTensor(ToTensorContext& opCtx)
{
    Pipeline pipeline(TO_TENSOR_THREAD_NUM);
    auto externalInput = AccDataOpSpec::Create("ExternalSource");
    if (!externalInput) {
        LogDebug << "Create ExternalSource specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }
    externalInput->AddOutput("ExternalSourceOutput", "cpu");
    auto toTensor = AccDataOpSpec::Create("ToTensor");
    if (!toTensor) {
        LogDebug << "Create ToTensor Operator specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }

    toTensor->AddInput("ExternalSourceOutput", "cpu");
    TensorLayout tensorLayout = ToTensorLayout(opCtx.format);
    toTensor->AddArg("layout", static_cast<int64_t>(tensorLayout));
    toTensor->AddOutput("ToTensorOutput", "cpu");

    ErrorCode ret = pipeline.Build({externalInput, toTensor}, "ToTensorOutput");
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