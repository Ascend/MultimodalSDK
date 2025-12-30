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
#ifndef ACCDATA_SRC_CPP_PIPELINE_GRAPH_GRAPH_H_
#define ACCDATA_SRC_CPP_PIPELINE_GRAPH_GRAPH_H_

#include <cinttypes>
#include <memory>
#include <vector>
#include <set>

#include "operator/operator.h"
#include "operator/op_spec.h"

namespace acclib {
namespace accdata {

using OpNodeId = uint32_t;
using DataNodeId = uint32_t;

struct OpNode {
    std::unique_ptr<Operator> op;
    std::set<OpNodeId> parents;
    std::set<OpNodeId> children;
    std::vector<DataNodeId> inputs;
    std::vector<DataNodeId> outputs;
};

struct DataNode {
    std::string name;
    OpNodeId producer;
    std::vector<OpNodeId> consumers;
};

class Graph {
public:
    /**
     * @brief Add graph node
     */
    AccDataErrorCode AddNode(const OpSpec &spec);

    /**
     * @brief Build based on desired outputs.
     *
     * @param [in] outputs      Output names.
     * @param [in] enableFusion Whether enable fusion or not
     */
    AccDataErrorCode Build(const std::vector<std::string> &outputs, bool enableFusion);

    uint64_t NumOpNode() const
    {
        return mOpNodes.size();
    }

    uint64_t NumDataNode() const
    {
        return mDataNodes.size();
    }

    /**
     * @brief Get the OpNode.
     *
     * @param [in] id   OpNode id, that is in [0, NumOpNode()).
     * @return const OpNode&
     */
    const OpNode &GetOpNode(OpNodeId id, AccDataErrorCode &errCode) const
    {
        errCode = AccDataErrorCode::H_PIPELINE_ERROR;
        if (id < NumOpNode()) {
            errCode = AccDataErrorCode::H_OK;
            return mOpNodes[id];
        }
        return opNode;
    }

    /**
     * @brief Get the DataNode.
     *
     * @param [in] id   OpNode id, that is in [0, NumDataNode()).
     * @return const DataNode&
     */
    const DataNode &GetDataNode(DataNodeId id, AccDataErrorCode &errCode) const
    {
        errCode = AccDataErrorCode::H_PIPELINE_ERROR;
        if (id < NumDataNode()) {
            errCode = AccDataErrorCode::H_OK;
            return mDataNodes[id];
        }
        return dataNode;
    }

    /**
     * @brief Get the DataNode ID by name.
     *
     * @param name
     * @return const DataNode&
     */
    AccDataErrorCode GetDataNodeId(const std::string &name, DataNodeId &dataNodeId) const
    {
        auto it = mName2DataNode.find(name);
        if (it == mName2DataNode.end()) {
            ACCDATA_ERROR("Name is not a DataNode.");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }
        dataNodeId = it->second;
        return AccDataErrorCode::H_OK;
    }

    /**
     * @brief Get the ids of DataNode that is specified as outputs during Build().
     */
    const std::vector<DataNodeId> &GetOutputs() const
    {
        return mOutputs;
    }

    /**
     * @brief Get stringed graph.
     */
    std::string ToString();

private:
    /**
     * Find the path produces the outputs
     * @param [in]  outputs       outputs names
     * @param [in]  throwIfErr    whether throw exception when encounter error or not.
     * @return a vector of OpNodeIds denote all the paths produce the outputs
     */
    AccDataErrorCode PathToOutputs(const std::vector<std::string> &outputs, std::vector<OpNodeId> &path);

    /**
     * @brief Fuse operators.
     */
    AccDataErrorCode Fuse(const std::vector<std::string> &outputs, std::vector<OpNodeId> &path);

    /**
     * Find fusion plan for the pipeline denoted by the parameter originPath
     * @param [in]  originPath  the path that generate the outputs
     * @param [in]  outputs     the name of all outputs
     * @return a vector of strings denotes the ops names in the found fusion plan
     */
    AccDataErrorCode FindFusePlan(const std::vector<OpNodeId> &originPath, const std::vector<std::string> &outputs,
                                  std::vector<std::string> &fusePlan);

    bool ValidateFusePlan(const std::vector<std::string> &plan, const std::vector<OpNodeId> &originPath,
        const std::vector<std::string> &outputs);

    AccDataErrorCode FuseOps(std::vector<OpNode> &opNodes, const std::string &fuseOpName,
        std::vector<OpNodeId> nodeIds);

    AccDataErrorCode UpdateGraph(std::vector<OpNode> &opNodes);

    /**
     * @brief Traverse the path that generate the output.
     *
     * @param [in]  output      Output name.
     * @param [out] path        Path of OpNodes that generates the specified output.
     */
    AccDataErrorCode Traverse(const std::string &output, std::vector<OpNodeId> &path);

    AccDataErrorCode BuildFromPath(const std::vector<OpNodeId> &path);

private:
    AccDataErrorCode LinkInput(const std::string &name, std::unordered_map<std::string, DataNodeId> &name2DataNode,
                               std::vector<DataNode> &dataNodes, size_t opNodeId, OpNode &opNode);

    std::vector<OpNode> mOpNodes{};
    std::vector<DataNode> mDataNodes{};
    std::unordered_map<std::string, DataNodeId> mName2DataNode{};
    std::vector<DataNodeId> mOutputs{};
    OpNode opNode{};
    DataNode dataNode{};
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_PIPELINE_GRAPH_GRAPH_H_
