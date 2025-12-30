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
 * Description: Processing of the Pipeline Api.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/core/framework/Pipeline.h"

#include <unistd.h>
#include <map>

#include "acc/utils/TensorUtils.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
using namespace acclib::accdata;
namespace {
using namespace Acc;
constexpr int DEFAULT_PIPELINE_BATCH_SIZE = 1;
constexpr int MIN_THREAD_NUM = 1;
constexpr int DEFAULT_PIPELINE_DEPTH = 2;
constexpr int SINGLE_INPUT_SIZE = 1;
constexpr int SINGLE_OUTPUT_SIZE = 1;
} // namespace

namespace Acc {
Pipeline::Pipeline(int numThreads, bool enableFusion)
{
    auto cpuCoreNum = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpuCoreNum == -1) {
        std::string msg = "Pipeline construct failed, because get the number of cpu core failed. It may be caused by "
                          "an internal system error or insufficient system resources, please check "
                          "the status of the system.";
        LogDebug << msg;
        throw std::runtime_error(msg);
    }
    if (numThreads < MIN_THREAD_NUM || numThreads > cpuCoreNum) {
        std::string msg = "Pipeline construct failed, because numThreads: " + std::to_string(numThreads) +
                          " is invalid, which should be in range [" + std::to_string(MIN_THREAD_NUM) + ", " +
                          std::to_string(cpuCoreNum) + "].";
        LogDebug << msg;
        throw std::runtime_error(msg);
    }
    pipeline_ = AccDataPipeline::Create(DEFAULT_PIPELINE_BATCH_SIZE, numThreads, DEFAULT_PIPELINE_DEPTH, enableFusion);
    if (!pipeline_) {
        std::string msg = "Pipeline construct failed, because create acc data pipeline resulted in nullptr. "
                          "It may be caused by out of memory, please check the memory status of the system.";
        LogDebug << msg;
        throw std::runtime_error(msg);
    }
}

ErrorCode Pipeline::Build(const std::vector<std::shared_ptr<AccDataOpSpec>>& specs, const std::string& output)
{
    // Currently only single output is supported. When calling the acc_data pipeline build interface,
    // it will be converted into a vector.
    auto ret = pipeline_->Build(specs, {output});
    if (ret != H_OK) {
        LogDebug << "Pipeline build failed, may be caused by a failure in processing acc data op spec or building "
                    "graph. Please check the input op specs and output op name."
                    << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
        return ERR_ACC_DATA_EXECUTE_FAILURE;
    }
    return SUCCESS;
}

ErrorCode Pipeline::Run(const std::unordered_map<std::string, std::vector<Tensor>>& inputs, Tensor& output, bool copy)
{
    // Currently only supports unordered map inputs of size 1.
    if (inputs.size() != SINGLE_INPUT_SIZE) {
        LogDebug << "Pipeline run failed, the parameter inputs size: " << inputs.size()
                 << " is invalid, currently only support single input." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    auto ret = SUCCESS;
    try {
        // convert inputs to accDataInputs
        std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> accDataInputs;
        AccDataErrorCode accDataRet = H_OK;
        auto input = inputs.begin();
        uint64_t tensorListSize = input->second.size();
        if (tensorListSize == 0) {
            LogDebug << "The vector size of inputs is zero, please check the inputs."
                     << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        auto tensorList = AccDataTensorList::Create(tensorListSize);
        if (tensorList == nullptr) {
            LogDebug << "acc_data tensor list create failed, may be caused by out of memory, please "
                        "check the memory status of the system." << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
            return ERR_ACC_DATA_EXECUTE_FAILURE;
        }
        for (size_t i = 0; i < tensorListSize; ++i) {
            auto tensorDataType = Acc::ToTensorDataType(input->second[i].DType());
            auto tensorLayout = Acc::ToTensorLayout(input->second[i].Format());
            std::shared_ptr<void> tensorSharedPtr(input->second[i].Ptr(), [](void*) {});
            accDataRet = tensorList->operator[](i).ShareData(tensorSharedPtr, input->second[i].Shape(), tensorDataType);
            if (accDataRet != H_OK) {
                LogDebug << "Copy inputs tensor vector index " << i << " to AccDataTensorList failed."
                    << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
                return ERR_ACC_DATA_EXECUTE_FAILURE;
            }
            tensorList->operator[](i).SetLayout(tensorLayout);
        }
        accDataInputs.insert(std::make_pair(input->first, tensorList));

        std::vector<std::shared_ptr<AccDataTensorList>> accDataOutputs;
        accDataRet = pipeline_->Run(accDataInputs, accDataOutputs, copy);
        if (accDataRet != H_OK) {
            LogDebug << "Pipeline run failed." << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
            return ERR_ACC_DATA_EXECUTE_FAILURE;
        }

        if (accDataOutputs.size() != SINGLE_OUTPUT_SIZE || accDataOutputs[0]->NumTensors() < SINGLE_OUTPUT_SIZE) {
            LogDebug << "Pipeline run result is not single output or tensor list size smaller than 1."
                << GetErrorInfo(ERR_ACC_DATA_EXECUTE_FAILURE);
            return ERR_ACC_DATA_EXECUTE_FAILURE;
        }
        // convert accDataOutputs to output, current only support single output
        auto tensorDataType = Acc::ToDataType(accDataOutputs[0]->operator[](0).DataType());
        auto tensorFormat = Acc::ToTensorFormat(accDataOutputs[0]->operator[](0).Layout());
        Tensor tensor(accDataOutputs[0]->operator[](0).RawDataPtr(), accDataOutputs[0]->operator[](0).Shape(),
                      tensorDataType, tensorFormat, "cpu");
        ret = tensor.Clone(output);
    } catch (const std::exception& e) {
        LogDebug << "Properties conversion between Multimodal SDK and acc_data tensors failed: " << e.what()
            << GetErrorInfo(ERR_ACC_DATA_PROPERTY_CONVERT_FAILURE);
        return ERR_ACC_DATA_PROPERTY_CONVERT_FAILURE;
    }
    return ret;
}
} // namespace Acc
