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
#ifndef ACCDATA_SRC_CPP_PIPELINE_EXECUTOR_EXECUTOR_H_
#define ACCDATA_SRC_CPP_PIPELINE_EXECUTOR_EXECUTOR_H_

#include <unistd.h>

#include "pipeline/graph/graph.h"
#include "pipeline/workspace/workspace.h"
#include "interface/accdata_error_code.h"

namespace acclib {
namespace accdata {

/**
 * @brief Executor base class
 */
class Executor {
public:
    Executor() = default;
    virtual ~Executor() = default;

    /**
     * @brief Set the maximum size of the batch that can be produced.
     *
     * @param [in] batchSize    Maximum size of the batch.
     */
    void SetBatchSize(int batchSize)
    {
        mBatchSize = batchSize;
    }

    /**
     * @brief Set the number of threads for executing operator.
     *
     * @param [in] numThreads   Number of threads.
     */
    void SetNumThreads(int numThreads)
    {
        mNumThreads = numThreads;
    }

    /**
     * @brief Set the depth of output queue.
     *
     * @param [in] depth        Queue depth.
     */
    void SetQueueDepth(int depth)
    {
        mQueueDepth = depth;
    }

    /**
     * @brief Feed external input.
     *
     * @param [in] name     Input name.
     * @param [in] data     Input data.
     * @param [in] copy     Copy the input date or not.
     */
    AccDataErrorCode FeedInput(const std::string &name, std::shared_ptr<TensorList> data, bool copy);

    /**
     * @brief Build by execution graph.
     *
     * @param [in] graph    Execution graph.
     */
    virtual AccDataErrorCode Build(Graph &&graph) = 0;

    /**
     * @brief Run the executor
     */
    virtual AccDataErrorCode Run() = 0;

    /**
     * @brief Get the generated outputs
     *
     * The routine is blocked and waits until outputs is ready. When it returns, the queue
     * position occupied by the outputs becomes available again.
     *
     * @note Run() must be called prior to calling this routine.
     * @param [out] ws      Workspace where outputs are saved.
     */
    virtual AccDataErrorCode Outputs(Workspace &ws) = 0;

protected:
    int mBatchSize { 1 };
    int mNumThreads { 1 };
    int mQueueDepth { 2 };
    Graph mGraph;
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_PIPELINE_EXECUTOR_EXECUTOR_H_
