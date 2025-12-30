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
 * @Date: 2025-2-18 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-18 17:00:00
 */
#include "accdata_pipeline_impl.h"

#include "executor/simple_executor.h"
#include "common/utility.h"

namespace acclib {
namespace accdata {
namespace {
constexpr int MIN_BATCH_SIZE = 1;
constexpr int MAX_BATCH_SIZE = 1024;
constexpr int MIN_THREAD_NUM = 1;
constexpr int MIN_QUEUE_DEPTH = 2;
constexpr int MAX_QUEUE_DEPTH = 128;
}
AccDataErrorCode CheckParams(int batchSize, int numThreads, int depth)
{
    if (batchSize < MIN_BATCH_SIZE || batchSize > MAX_BATCH_SIZE) {
        ACCDATA_ERROR("BatchSize(" << batchSize << ") should in [1, 1024]");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }
    auto coreNum = sysconf(_SC_NPROCESSORS_ONLN);
    if (numThreads < MIN_THREAD_NUM || numThreads > coreNum) {
        ACCDATA_ERROR("NumThreads(" << numThreads << ") should in [1, " << coreNum << "]");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }
    if (depth < MIN_QUEUE_DEPTH || depth > MAX_QUEUE_DEPTH) {
        ACCDATA_ERROR("depth(" << depth << ") should in [2, 128]");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }
    return AccDataErrorCode::H_OK;
}

std::shared_ptr<AccDataPipeline> AccDataPipeline::Create(int batchSize, int numThreads, int depth,
    bool enableFusion)
{
    AccDataErrorCode errCode = CheckParams(batchSize, numThreads, depth);
    if (errCode != AccDataErrorCode::H_OK) {
        return nullptr;
    }
    return std::make_shared<AccDataPipelineImpl>(batchSize, numThreads, depth, enableFusion);
}

AccDataPipelineImpl::AccDataPipelineImpl(int batchSize, int numThreads, int depth, bool enableFusion)
    : mEnableFusion(enableFusion)
{
    mExecutor = std::make_unique<SimpleExecutor>();
    mExecutor->SetBatchSize(batchSize);
    mExecutor->SetNumThreads(numThreads);
    mExecutor->SetQueueDepth(depth);
    ACCDATA_INFO("success to build pipeline(batchSize: " << batchSize << ", threadCnt: " << numThreads << ", depth: "
        << depth << ", enableFusion: " << enableFusion << ")");
}

AccDataPipelineImpl::~AccDataPipelineImpl()
{
    ACCDATA_INFO("pipeline destructed");
}

AccDataErrorCode AccDataPipelineImpl::Build(const std::vector<std::shared_ptr<AccDataOpSpec>> &specs,
    const std::vector<std::string> &outputs)
{
    auto errCode = AccDataErrorCode::H_OK;
    Graph graph;
    for (auto spec : specs) {
        std::shared_ptr<OpSpec> op_spec = std::dynamic_pointer_cast<OpSpec>(spec);
        ACCDATA_CHECK_ERRORCODE_RETURN(op_spec.get() != nullptr, "Failed to dynamic cast spec.",
            AccDataErrorCode::H_COMMON_NULLPTR);
        errCode = graph.AddNode(*op_spec);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to add node.", errCode);
    }
    errCode = graph.Build(outputs, mEnableFusion);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to build graph.", errCode);
    ACCDATA_DEBUG(graph.ToString());
    return mExecutor->Build(std::move(graph));
}

AccDataErrorCode AccDataPipelineImpl::Run(std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> inputs,
    std::vector<std::shared_ptr<AccDataTensorList>>& opOutputs, bool copy)
{
    auto errorCode = AccDataErrorCode::H_OK;
    for (auto input : inputs) {
        std::shared_ptr<TensorList> tensorList = std::dynamic_pointer_cast<TensorList>(input.second);
        ACCDATA_CHECK_ERRORCODE_RETURN(tensorList.get() != nullptr, "Invalid input tensorList.",
            AccDataErrorCode::H_COMMON_NULLPTR);
        errorCode = mExecutor->FeedInput(input.first, tensorList, copy);
        ACCDATA_CHECK_ERRORCODE_RETURN(errorCode == AccDataErrorCode::H_OK, "Failed to feed input data.", errorCode);
    }

    errorCode = mExecutor->Run();
    ACCDATA_CHECK_ERRORCODE_RETURN(errorCode == AccDataErrorCode::H_OK, "Failed to run ops.", errorCode);
    Workspace ws;
    errorCode = Outputs(ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errorCode == AccDataErrorCode::H_OK, "Failed to get pipeline outputs.", errorCode);

    std::vector<std::shared_ptr<AccDataTensorList>> tmpOpOutputs;
    tmpOpOutputs.resize(ws.NumOutput());

    std::shared_ptr<TensorList> outputPtr;

    for (uint64_t i = 0; i < ws.NumOutput(); ++i) {
        errorCode = ws.GetOutputPtr(i, outputPtr);
        ACCDATA_CHECK_ERRORCODE_RETURN(errorCode == AccDataErrorCode::H_OK, "Failed to get output ptr.", errorCode);
        tmpOpOutputs[i] = outputPtr;
    }

    opOutputs.resize(ws.NumOutput());
    opOutputs = tmpOpOutputs;

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode AccDataPipelineImpl::Outputs(Workspace &ws)
{
    return mExecutor->Outputs(ws);
}

} // namespace accdata
} // namespace acclib
