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
#ifndef ACCDATA_SRC_CPP_PIPELINE_WORKSPACE_WORKSPACE_MANAGER_H_
#define ACCDATA_SRC_CPP_PIPELINE_WORKSPACE_WORKSPACE_MANAGER_H_

#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "workspace.h"
#include "pipeline/graph/graph.h"
#include "common/thread_pool.h"

namespace acclib {
namespace accdata {

/**
 * @brief Workspace Manager
 *
 *
 */
class WorkspaceManager {
public:
    using StoreQueue = std::vector<std::shared_ptr<TensorList>>;
    using WorkspaceQueue = std::vector<Workspace>;
    using QueueIdx = int32_t;
    const static QueueIdx gInvalidQueueIdx = -1;

public:
    WorkspaceManager() = default;
    ~WorkspaceManager() = default;

    /**
     * @brief Initialize workspaces for graph nodes.
     *
     * Initialize enough workspaces and store to execute N times pipeline and save the outputs. N is
     * the maximum of @b queueDepth. If queueDepth[i] is less than N, the first store of DataNode is
     * reused in the workspace after the i-th pipeline execution.
     *
     * @param [in] graph            Pipeline graph.
     * @param [in] queueDepth       Each element indicates the depth of a DataNode store queue.
     * @param [in] maxBatchSize     Maximum size of the batch that can be produced.
     * @param [in] pool             Thread pool for executing OpNode.
     *
     * @return AccDataErrorCode     Runs successfully when it is AccDataErrorCode::H_OK
     */
    AccDataErrorCode Init(const Graph &graph, const std::vector<int> &queueDepth, int maxBatchSize,
              std::shared_ptr<ThreadPool> pool);

    /**
     * @brief TryAcquire a free workspace queue to run pipeline.
     *
     * It is called before executing graph to obtain available workspace queue index. It blocks until
     * an available index is obtained or timeout.
     */
    QueueIdx TryAcquireFreeIdx();

    /**
     * @brief Acquire a free workspace queue to run pipeline.
     *
     * It is called before executing graph to obtain available workspace queue index. It blocks until
     * an available index is obtained.
     */
    QueueIdx Acquire();

    /**
     * @brief Release the acquired queue.
     *
     * Once released, It means the pipeline output is ready in the queue. The ready queue can be
     * obtained through AcquireOutputIdx() and can not be Acquire() again.
     *
     * @param [in] idx      Queue index returned by Acquire().
     */
    void Release(QueueIdx idx);

    /**
     * @brief Try Acquire a ready queue.
     *
     * The pipeline output is ready in the corresponding workspace.
     */
    QueueIdx TryAcquireOutputIdx();

    /**
     * @brief Acquire a ready queue.
     *
     * The pipeline output is ready in the corresponding workspace.
     */
    QueueIdx AcquireOutputIdx();

    /**
     * @brief Release the last QueueIdx returned by AcquireOutputIdx().
     *
     * Once released, the queue can be Acquire() again.
     */
    void ReleaseOutputIdx();

    /**
     * @brief Recycle the queue.
     *
     * The recycled queue becomes a free queue and can be obtained through Acquire() again.
     *
     * @param [in] idx  Queue index.
     */
    void Recycle(QueueIdx idx);

    /**
     * @brief Get the OpNode workspace.
     *
     * @param [in] queueIdx     Queue index returned by Acquire().
     * @param [in] opNodeId     OpNode ID.
     * @return Workspace&
     */
    Workspace& GetWorkspace(QueueIdx queueIdx, OpNodeId opNodeId);

    /**
     * @brief Get the data store.
     *
     * @param [in] queueIdx     Queue index.
     * @param [in] dataNodeId   DataNode ID.
     * @return std::shared_ptr<TensorList>
     */
    const std::shared_ptr<TensorList>& GetDataStore(QueueIdx queueIdx, DataNodeId dataNodeId);

private:
    AccDataErrorCode InitStoreQueue(const Graph &graph, const std::vector<int> &queueDepth, int maxBatchSize);

    AccDataErrorCode InitWorkspaceQueue(const Graph &graph, std::shared_ptr<ThreadPool> pool);

    AccDataErrorCode InitOneWorkspaceQueue(const Graph &graph, std::shared_ptr<ThreadPool> pool, QueueIdx idx);

    AccDataErrorCode InitWorkspace(const Graph &graph, std::shared_ptr<ThreadPool> pool,
        QueueIdx queueIdx, OpNodeId opNodeId);

private:
    int32_t mMaxQueueDepth { 0 };
    /* Each DataNode has a StoreQueue. */
    std::vector<StoreQueue> mStoreQueues{};
    /* Each pipeline run has a WorkspaceQueue. */
    std::vector<WorkspaceQueue> mWorkspaceQueues{};

    /* Free queue index. */
    std::mutex mFreeMutex{};
    std::condition_variable mFreeCond{};
    std::queue<QueueIdx> mFreeQueueIdx{};

    /* Data Ready Queue Index. */
    std::mutex mReadyMutex{};
    std::condition_variable mReadyCond{};
    std::queue<QueueIdx> mReadyQueueIdx{};

    /* Externally shared queue index. */
    std::mutex mShareMutex{};
    std::queue<QueueIdx> mShareQueueIdx{};

    int32_t mWaitTime { 200 }; // wait 200 ms
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_PIPELINE_WORKSPACE_WORKSPACE_MANAGER_H_
