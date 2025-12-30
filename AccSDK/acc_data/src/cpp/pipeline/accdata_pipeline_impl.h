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
#ifndef ACCDATA_SRC_CPP_PIPELINE_IMPL_H_
#define ACCDATA_SRC_CPP_PIPELINE_IMPL_H_

#include <vector>
#include <memory>

#include "operator/op_spec.h"
#include "workspace/workspace.h"
#include "executor/executor.h"
#include "accdata_error_code.h"
#include "accdata_pipeline.h"

namespace acclib {
namespace accdata {
/**
 * @brief Define the data processing workflow.
 */
class AccDataPipelineImpl : public AccDataPipeline {
public:
    AccDataPipelineImpl(int batchSize, int numThreads, int depth, bool enableFusion = true);
    virtual ~AccDataPipelineImpl();

public:

    /**
     * @brief Build a pipeline for generating specified outputs.
     *
     * @param [in] specs     Operator specification.
     * @param [in] outputs      Desired output names
     *
     * @return AccDataErrorCode Runs successfully when it is AccDataErrorCode::H_OK
     */
    AccDataErrorCode Build(const std::vector<std::shared_ptr<AccDataOpSpec>> &specs,
        const std::vector<std::string> &outputs);

    /**
     * @brief Run the pipeline
     */
    AccDataErrorCode Run(std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> inputs,
        std::vector<std::shared_ptr<AccDataTensorList>>& opOutputs, bool copy);

    /**
     * @brief Get the generated outputs
     *
     * The routine is blocked and waits until outputs is ready. When it returns, the last workspace
     * obtained through Outputs() will be used again.
     *
     * @note Run() must be called prior to calling this routine.
     *
     * @param [out] ws      Workspace where outputs are saved.
     */
    AccDataErrorCode Outputs(Workspace &ws);

private:
    std::unique_ptr<Executor> mExecutor;
    bool mEnableFusion { false };
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_PIPELINE_IMPL_H_
