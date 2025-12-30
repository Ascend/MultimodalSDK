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
#ifndef ACCDATA_SRC_CPP_PIPELINE_EXECUTOR_SIMPLE_EXECUTOR_H_
#define ACCDATA_SRC_CPP_PIPELINE_EXECUTOR_SIMPLE_EXECUTOR_H_

#include "common/thread_pool.h"
#include "executor.h"
#include "pipeline/workspace/workspace_manager.h"

namespace acclib {
namespace accdata {

class SimpleExecutor : public Executor {
public:
    enum class State {
        INIT,
        BUILT,
        ERROR,
    };

public:
    AccDataErrorCode Build(Graph &&graph) override;

    AccDataErrorCode Run() override;

    AccDataErrorCode Outputs(Workspace &ws) override;

private:
    State mState { State::INIT };
    std::shared_ptr<ThreadPool> mThreadPool;
    WorkspaceManager mWorkspaceManager;
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_PIPELINE_EXECUTOR_SIMPLE_EXECUTOR_H_
