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
#include "executor.h"

#include "operator/input/external_source.h"

namespace acclib {
namespace accdata {

AccDataErrorCode Executor::FeedInput(const std::string &name, std::shared_ptr<TensorList> data, bool copy)
{
    ACCDATA_DEBUG("Feed input '" << name << "'.");

    auto feedInputHelper = [name, data, copy, this]() -> AccDataErrorCode {
        auto errCode = AccDataErrorCode::H_OK;
        DataNodeId dataNodeId;
        errCode = mGraph.GetDataNodeId(name, dataNodeId);
        if (errCode != AccDataErrorCode::H_OK) {
            return errCode;
        }
    
        auto &dataNode = mGraph.GetDataNode(dataNodeId, errCode);
        if (errCode != AccDataErrorCode::H_OK) {
            return errCode;
        }
    
        auto &opNode = mGraph.GetOpNode(dataNode.producer, errCode);
        if (errCode != AccDataErrorCode::H_OK) {
            return errCode;
        }

        if (opNode.op == nullptr || opNode.op->GetSpec().Name() != "ExternalSource") {
            ACCDATA_ERROR("Feed input key must be ExternalSource operator.");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }

        auto *op = dynamic_cast<ExternalSource*>(opNode.op.get());
        if (op == nullptr) {
            ACCDATA_ERROR("Feed input key cast ExternalSource fail.");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }
        errCode = op->Feed(data, copy);
    
        return errCode;
    };

    auto errCode = feedInputHelper();
    if (errCode != AccDataErrorCode::H_OK) {
        ACCDATA_WARN("Failed to feed DataNode. May be useless external input which is removed.");
    }
    return AccDataErrorCode::H_OK;
}

} // namespace accdata
} // namespace acclib
