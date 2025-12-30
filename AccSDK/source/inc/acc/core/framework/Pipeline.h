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
* Description: Definition of Pipeline API.
* Author: ACC SDK
* Create: 2025
* History: NA
*/

#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "accdata_pipeline.h"
#include "accdata_op_spec.h"
#include "acc/tensor/Tensor.h"

namespace Acc {
class Pipeline {
public:
    /**
     * @brief Construct a new Pipeline object
     *
     * @param numThreads The number of threads running in parallel, ranging from [1, number of CPU cores]
     * @param enableFusion Whether to enable cpu operator fusion
     */
    explicit Pipeline(int numThreads = 1, bool enableFusion = true);

    /**
     * @brief Destroy a Pipeline object
     *
     */
    ~Pipeline() = default;

    /**
     * @brief Build the pipeline based on operator specification and output
     *
     * @param specs Operator Specification list
     * @param output The output name of the pipeline
     */
    ErrorCode Build(const std::vector<std::shared_ptr<acclib::accdata::AccDataOpSpec>> &specs,
                    const std::string &output);

    /**
     * @brief Build the pipeline based on operator specification and output
     *
     * @param inputs The input operator name and corresponding input tensor of pipeline
     * @param output The output tensor
     * @param copy The pipeline copies the input or not, i.e., whether the memory is shared
     */
    ErrorCode Run(const std::unordered_map<std::string, std::vector<Tensor>> &inputs, Tensor &output, bool copy);

private:
    std::shared_ptr<acclib::accdata::AccDataPipeline> pipeline_;
};
} // namespace Acc
#endif // PIPELINE_H
