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
 * Description: QwenFusionOperator op on cpu.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/CPUAccelerator.h"
#include "acc/core/framework/Pipeline.h"
#include "acc/tensor/TensorOps.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
#include "acc/utils/TensorUtils.h"
#include "accdata_tensor.h"
#include "accdata_op_spec.h"

using namespace acclib::accdata;
namespace {
constexpr size_t DEFAULT_QWEN_FUSION_THREAD_NUM = 8;
using namespace Acc;
/**
 * @brief Build preprocessing pipeline for QwenFusion
 *
 * Constructs ToTensor -> Normalize sequence
 */
ErrorCode BuildPreprocessQwenPipeline(Pipeline& pipeline, const std::vector<float>& mean, const std::vector<float>& std,
                                      TensorFormat layout)
{
    auto externalInput = acclib::accdata::AccDataOpSpec::Create("ExternalSource");
    if (!externalInput) {
        LogDebug << "Create ExternalSource specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }
    externalInput->AddOutput("ExternalSourceOutput", "cpu");

    auto toTensorOp = acclib::accdata::AccDataOpSpec::Create("ToTensor");
    if (!toTensorOp) {
        LogDebug << "Create ToTensor Operator specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }
    TensorLayout tensorLayout = ToTensorLayout(layout);
    toTensorOp->AddInput("ExternalSourceOutput", "cpu");
    toTensorOp->AddArg("layout", static_cast<int64_t>(tensorLayout));
    toTensorOp->AddOutput("TensorOutput", "cpu");

    auto normalizeOp = acclib::accdata::AccDataOpSpec::Create("Normalize");
    if (!normalizeOp) {
        LogDebug << "Create Normalize Operator specification failed, please set correct operator name in acc data."
                 << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }
    normalizeOp->AddInput("TensorOutput", "cpu");
    normalizeOp->AddArg("mean", mean);
    normalizeOp->AddArg("stddev", std);
    normalizeOp->AddOutput("NormalizedOutput", "cpu");

    return pipeline.Build({externalInput, toTensorOp, normalizeOp}, "NormalizedOutput");
}
} // namespace
namespace Acc {
ErrorCode CPUAccelerator::QwenFusionOperator(QwenFusionContext& opCtx)
{
    Pipeline pipeline(DEFAULT_QWEN_FUSION_THREAD_NUM);
    ErrorCode ret = BuildPreprocessQwenPipeline(pipeline, opCtx.mean, opCtx.std, opCtx.layout);
    if (ret != SUCCESS) {
        LogError << "Failed to build preprocessing pipeline" << GetErrorInfo(ret);
        return ret;
    }
    size_t numInputs = opCtx.inputTensorRefs.size();
    for (size_t i = 0; i < numInputs; ++i) {
        const Tensor& src = opCtx.inputTensorRefs[i].get();
        Tensor& dst = opCtx.outputTensorRefs[i].get();

        // Resize tensor
        ret = TensorResize(src, dst, opCtx.resizeH, opCtx.resizeW, Interpolation::BICUBIC, opCtx.deviceMode);
        if (ret != SUCCESS) {
            LogError << "Tensor resize failed for input " << i << GetErrorInfo(ret);
            return ret;
        }

        // Run pipeline (ToTensor + Normalize)
        std::unordered_map<std::string, std::vector<Tensor>> inputs;
        inputs["ExternalSourceOutput"].push_back(dst);
        ret = pipeline.Run(inputs, dst, true);
        if (ret != SUCCESS) {
            LogError << "Pipeline run failed for input " << i << GetErrorInfo(ret);
            return ret;
        }
    }

    return SUCCESS;
}
} // namespace Acc