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
 * @Date: 2025-2-11 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-11 17:00:00
 */

#ifndef ACCDATA_SRC_CPP_PIPELINE_WORKSPACE_WORKSPACE_H_
#define ACCDATA_SRC_CPP_PIPELINE_WORKSPACE_WORKSPACE_H_

#include <unordered_map>

#include "common/check.h"
#include "common/thread_pool.h"
#include "accdata_error_code.h"
#include "tensor/tensor_list.h"

namespace acclib {
namespace accdata {

/**
 * @brief Operator execution context.
 */
class Workspace {
public:
    Workspace() = default;
    ~Workspace() = default;

    void Clear()
    {
        mOpArgInputs.clear();
        mOpArgInputIdxs.clear();
        mOpInputs.clear();
        mOpOutputs.clear();
        return;
    }

    void SetThreadPool(std::shared_ptr<ThreadPool> pool)
    {
        mPool = pool;
        return;
    }

    void AddInput(std::shared_ptr<TensorList> input)
    {
        mOpInputs.push_back(input);
        return;
    }

    void AddOutput(std::shared_ptr<TensorList> output)
    {
        mOpOutputs.push_back(output);
        return;
    }

    /**
     * @brief Add argument input.
     *
     * @param [in] name     Input name.
     * @param [in] input    Input tensor.
     */
    void AddArgInput(const std::string& name, std::shared_ptr<TensorList> input)
    {
        mOpArgInputIdxs[name] = mOpArgInputs.size();
        mOpArgInputs.push_back(input);
        return;
    }

    uint64_t NumInput() const
    {
        return mOpInputs.size();
    }

    uint64_t NumOutput() const
    {
        return mOpOutputs.size();
    }

    uint64_t NumArgInput() const
    {
        return mOpArgInputs.size();
    }

    ThreadPool& GetThreadPool() const
    {
        return *mPool;
    }

    inline AccDataErrorCode GetNumThreads(int &threadNum) const
    {
        if (mPool == nullptr) {
            ACCDATA_ERROR("The mPool is nullptr");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }
        threadNum = mPool->NumThreads();
        return AccDataErrorCode::H_OK;
    }

    /**
     * @brief Get input tensors.
     */
    const TensorList& GetInput(uint64_t idx, AccDataErrorCode &errCode) const
    {
        errCode = AccDataErrorCode::H_PIPELINE_ERROR;
        if (idx < NumInput()) {
            errCode = AccDataErrorCode::H_OK;
            return *mOpInputs[idx];
        }
        return *tensorList;
    }

    /**
     * @brief Get output tensors.
     */
    TensorList& GetOutput(uint64_t idx, AccDataErrorCode &errCode) const
    {
        errCode = AccDataErrorCode::H_PIPELINE_ERROR;
        if (idx < NumOutput()) {
            errCode = AccDataErrorCode::H_OK;
            return *mOpOutputs[idx];
        }
        return *tensorList;
    }

    /**
     * @brief Get output tensors.
     */
    AccDataErrorCode GetOutputPtr(uint64_t idx, std::shared_ptr<TensorList> &outputPtr) const
    {
        if (idx >= NumOutput()) {
            ACCDATA_ERROR("Out of range.");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }
        outputPtr = mOpOutputs[idx];
        return AccDataErrorCode::H_OK;
    }

    /**
     * @brief Get argument input tensors.
     *
     * @param [in] name     Intput name.
     */
    const TensorList& GetArgInput(const std::string& name, AccDataErrorCode &errCode) const
    {
        errCode = AccDataErrorCode::H_PIPELINE_ERROR;
        auto it = mOpArgInputIdxs.find(name);
        if (it != mOpArgInputIdxs.end()) {
            errCode = AccDataErrorCode::H_OK;
            return *mOpArgInputs[it->second];
        }
        return *tensorList;
    }

private:
    std::shared_ptr<ThreadPool> mPool{};
    /* Argument initialized by input */
    std::vector<std::shared_ptr<TensorList>> mOpArgInputs{};
    std::unordered_map<std::string, int> mOpArgInputIdxs{};
    /* Input and Output */
    std::vector<std::shared_ptr<TensorList>> mOpInputs{};
    std::vector<std::shared_ptr<TensorList>> mOpOutputs{};
    std::shared_ptr<TensorList> tensorList{ nullptr };
};

} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_PIPELINE_WORKSPACE_WORKSPACE_H_
