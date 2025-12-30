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
#include "workspace_manager.h"

namespace acclib {
namespace accdata {

using QueueIdx = WorkspaceManager::QueueIdx;

AccDataErrorCode WorkspaceManager::Init(const Graph &graph, const std::vector<int> &queueDepth,
    int maxBatchSize, std::shared_ptr<ThreadPool> pool)
{
    auto errCode = AccDataErrorCode::H_OK;
    errCode = InitStoreQueue(graph, queueDepth, maxBatchSize);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to init store queue.", errCode);
    errCode = InitWorkspaceQueue(graph, pool);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to init workspace.", errCode);
    /* All is free. */
    for (int i = 0; i < mMaxQueueDepth; ++i) {
        mFreeQueueIdx.push(i);
    }
    return AccDataErrorCode::H_OK;
}

QueueIdx WorkspaceManager::TryAcquireFreeIdx()
{
    std::unique_lock<std::mutex> lock(mFreeMutex);

    mFreeCond.wait_for(lock, std::chrono::milliseconds(mWaitTime), [this]() { return !mFreeQueueIdx.empty(); });
    if (mFreeQueueIdx.empty()) {
        return gInvalidQueueIdx;
    }
    auto idx = mFreeQueueIdx.front();
    mFreeQueueIdx.pop();
    return idx;
}

QueueIdx WorkspaceManager::Acquire()
{
    std::unique_lock<std::mutex> lock(mFreeMutex);
    mFreeCond.wait(lock, [this]() {return !mFreeQueueIdx.empty();});
    auto idx = mFreeQueueIdx.front();
    mFreeQueueIdx.pop();
    return idx;
}

void WorkspaceManager::Release(QueueIdx idx)
{
    {
        std::unique_lock<std::mutex> lock(mReadyMutex);
        mReadyQueueIdx.push(idx);
    }
    mReadyCond.notify_one();
    return;
}

QueueIdx WorkspaceManager::TryAcquireOutputIdx()
{
    QueueIdx idx = -1;
    {
        std::unique_lock<std::mutex> lock(mReadyMutex);
        mReadyCond.wait_for(lock, std::chrono::milliseconds(mWaitTime), [this]() {return !mReadyQueueIdx.empty();});
        if (mReadyQueueIdx.empty()) {
            ACCDATA_ERROR("Accdata get reday queue index error");
            return gInvalidQueueIdx;
        }
        idx = mReadyQueueIdx.front();
        mReadyQueueIdx.pop();
    }
    std::lock_guard<std::mutex> lock(mShareMutex);
    mShareQueueIdx.push(idx);
    return idx;
}

QueueIdx WorkspaceManager::AcquireOutputIdx()
{
    QueueIdx idx = -1;
    {
        std::unique_lock<std::mutex> lock(mReadyMutex);
        mReadyCond.wait(lock, [this]() {return !mReadyQueueIdx.empty();});
        idx = mReadyQueueIdx.front();
        mReadyQueueIdx.pop();
    }
    std::lock_guard<std::mutex> lock(mShareMutex);
    mShareQueueIdx.push(idx);
    return idx;
}

void WorkspaceManager::ReleaseOutputIdx()
{
    if (!mShareQueueIdx.empty()) {
        QueueIdx idx = -1;
        {
            std::lock_guard<std::mutex> lock(mShareMutex);
            if (mShareQueueIdx.empty()) {
                return;
            }
            idx = mShareQueueIdx.front();
            mShareQueueIdx.pop();
        }
        Recycle(idx);
    }
    return;
}

void WorkspaceManager::Recycle(QueueIdx idx)
{
    {
        std::lock_guard<std::mutex> lock(mFreeMutex);
        mFreeQueueIdx.push(idx);
    }
    mFreeCond.notify_one();
    return;
}

Workspace& WorkspaceManager::GetWorkspace(QueueIdx queueIdx, OpNodeId opNodeId)
{
    return mWorkspaceQueues[queueIdx][opNodeId];
}

const std::shared_ptr<TensorList>& WorkspaceManager::GetDataStore(QueueIdx queueIdx, DataNodeId dataNodeId)
{
    auto &storeQueue = mStoreQueues[dataNodeId];
    if (queueIdx >= int(storeQueue.size())) {
        return storeQueue[0];
    }
    return storeQueue[queueIdx];
}

AccDataErrorCode WorkspaceManager::InitStoreQueue(const Graph &graph, const std::vector<int> &queueDepth,
    int maxBatchSize)
{
    int numDataNode = graph.NumDataNode();
    if (numDataNode != int(queueDepth.size())) {
        ACCDATA_ERROR("Each DataNode should have a queue depth.");
        return AccDataErrorCode::H_PIPELINE_ERROR;
    }
    mStoreQueues.resize(numDataNode);
    for (int i = 0; i < numDataNode; ++i) {
        auto depth = queueDepth[i];
        if (depth > mMaxQueueDepth) {
            mMaxQueueDepth = depth;
        }
        if (depth <= 0) {
            ACCDATA_ERROR("Queue depth should be greater than 0.");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }
        auto &storeQueue = mStoreQueues[i];
        storeQueue.resize(depth);
        for (int j = 0; j < depth; ++j) {
            storeQueue[j] = std::make_shared<TensorList>(maxBatchSize);
        }
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode WorkspaceManager::InitWorkspaceQueue(const Graph &graph, std::shared_ptr<ThreadPool> pool)
{
    /* mMaxQueueDepth workspace queues are required to generate mMaxQueueDepth pipeline outputs. */
    mWorkspaceQueues.resize(mMaxQueueDepth);
    for (int i = 0; i < mMaxQueueDepth; ++i) {
        auto errCode = InitOneWorkspaceQueue(graph, pool, i);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to init one workspace queue.",
                                       errCode);
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode WorkspaceManager::InitOneWorkspaceQueue(const Graph &graph, std::shared_ptr<ThreadPool> pool,
    QueueIdx idx)
{
    auto &queue = mWorkspaceQueues[idx];
    /* Each OpNode has a workspace. */
    int numOpNode = graph.NumOpNode();
    queue.resize(numOpNode);
    for (int i = 0; i < numOpNode; ++i) {
        auto errCode = InitWorkspace(graph, pool, idx, i);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to init workspace.", errCode);
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode WorkspaceManager::InitWorkspace(const Graph &graph, std::shared_ptr<ThreadPool> pool,
    QueueIdx queueIdx, OpNodeId opNodeId)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto &ws = GetWorkspace(queueIdx, opNodeId);
    ws.SetThreadPool(pool);
    auto &opNode = graph.GetOpNode(opNodeId, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);

    const auto &spec = opNode.op->GetSpec();
    OpSpec::InOutDesc input;
    DataNodeId dataNodeId;
    OpSpec::InOutDesc output;
    /* Add inputs. */
    for (uint64_t i = 0; i < spec.NumInput(); ++i) {
        errCode = spec.GetInput(i, input);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get input.", errCode);
        auto inputName = input.name;
        errCode = graph.GetDataNodeId(inputName, dataNodeId);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get data node id", errCode);
        auto &input = GetDataStore(queueIdx, dataNodeId);
        ws.AddInput(input);
    }
    /* Add argument inputs. */
    std::string inputName;
    for (auto &args : spec.GetArgInputIdxs()) {
        errCode = spec.GetArgInput(args.second, inputName);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument input.", errCode);
        errCode = graph.GetDataNodeId(inputName, dataNodeId);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get data node id", errCode);
        auto &input = GetDataStore(queueIdx, dataNodeId);
        ws.AddArgInput(args.first, input);
    }

    /* Add outputs. */
    for (uint64_t i = 0; i < spec.NumOutput(); ++i) {
        errCode = spec.GetOutput(i, output);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get output.", errCode);
        auto outputName = output.name;
        errCode = graph.GetDataNodeId(outputName, dataNodeId);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get data node id", errCode);
        auto &output = GetDataStore(queueIdx, dataNodeId);
        ws.AddOutput(output);
    }
    return AccDataErrorCode::H_OK;
}

} // namespace accdata
} // namespace acclib
