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
 * @Date: 2025-2-17 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-17 17:00:00
 */
#include "simple_executor.h"

namespace acclib {
namespace accdata {

AccDataErrorCode SimpleExecutor::Build(Graph &&graph)
{
    if (mState != State::INIT) {
        ACCDATA_ERROR("Unexpected state.");
        return AccDataErrorCode::H_PIPELINE_STATE_ERROR;
    }
    if (mNumThreads <= 0) {
        ACCDATA_ERROR("Number of threads must be greater than 0.");
        return AccDataErrorCode::H_PIPELINE_BUILD_ERROR;
    }
    mThreadPool = std::make_shared<ThreadPool>(mNumThreads, false, "AccData");
    mGraph = std::move(graph);
    /* Initialize queue depth. Operators are executed in sequence. Therefore, the DataNode depth of
    non-pipeline output only needs to be 1. */
    std::vector<int> queueDepth;
    queueDepth.resize(mGraph.NumDataNode(), 1);
    for (auto id : mGraph.GetOutputs()) {
        queueDepth[id] = mQueueDepth;
    }
    mWorkspaceManager.Init(mGraph, queueDepth, mBatchSize, mThreadPool);
    mState = State::BUILT;
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode SimpleExecutor::Run()
{
    if (mState != State::BUILT) {
        ACCDATA_ERROR("Unexpected state.");
        return AccDataErrorCode::H_PIPELINE_STATE_ERROR;
    }

    auto errCode = AccDataErrorCode::H_OK;
    auto idx = mWorkspaceManager.TryAcquireFreeIdx();
    if (idx == WorkspaceManager::gInvalidQueueIdx) {
        ACCDATA_ERROR("Accdata acquire FreeIdx timeout!");
        return AccDataErrorCode::H_PIPELINE_ERROR;
    }

    for (uint64_t i = 0; i < mGraph.NumOpNode(); ++i) {
        const auto &opNode = mGraph.GetOpNode(i, errCode);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);
        auto &ws = mWorkspaceManager.GetWorkspace(idx, i);
        errCode = opNode.op->Run(ws);
        if (errCode != AccDataErrorCode::H_OK) {
            ACCDATA_ERROR("Accdata executor run " << (opNode.op->GetSpec().Name()) << " error :" << errCode);
            mWorkspaceManager.Recycle(idx);
            mState = State::ERROR;
            return errCode;
        }
    }

    mWorkspaceManager.Release(idx);
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode SimpleExecutor::Outputs(Workspace &ws)
{
    if (mState != State::BUILT) {
        ACCDATA_ERROR("Unexpected state.");
        return AccDataErrorCode::H_PIPELINE_STATE_ERROR;
    }
    /* Release the last shared data first. */
    mWorkspaceManager.ReleaseOutputIdx();
    /* Get the current outputs. */
    auto idx = mWorkspaceManager.TryAcquireOutputIdx();
    if (idx == WorkspaceManager::gInvalidQueueIdx) {
        return AccDataErrorCode::H_PIPELINE_ERROR;
    }
    for (auto id : mGraph.GetOutputs()) {
        auto &data = mWorkspaceManager.GetDataStore(idx, id);
        ws.AddOutput(data);
    }
    return AccDataErrorCode::H_OK;
}

} // namespace accdata
} // namespace acclib
