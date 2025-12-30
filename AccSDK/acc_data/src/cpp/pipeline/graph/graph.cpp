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
#include "graph.h"

#include "common/string_util.h"
#include "common/tracer.h"
#include "operator/op_factory.h"

namespace acclib {
namespace accdata {

AccDataErrorCode Graph::AddNode(const OpSpec &spec)
{
    auto errCode = AccDataErrorCode::H_OK;
    /* Only add node, not build the relationship between nodes. */
    OpNode opNode;
    errCode = OpFactory::Instance().Create(spec.Name(), spec, opNode.op);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to create op factory instance.", errCode);
    OpNodeId opNodeId = mOpNodes.size();

    std::unordered_map<std::string, DataNodeId> inputName2DataNode;
    std::vector<DataNode> inputDataNodes;

    for (uint64_t i = 0; i < spec.NumOutput(); ++i) {
        OpSpec::InOutDesc output;
        errCode = spec.GetOutput(i, output);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get output.", errCode);
        auto name = output.name;
        if (mName2DataNode.count(name) != 0 || inputName2DataNode.count(name) != 0) {
            ACCDATA_ERROR("The DataNode already existed.");
            return AccDataErrorCode::H_PIPELINE_ERROR;
        }

        inputName2DataNode[name] = mDataNodes.size() + inputDataNodes.size();
        auto &dataNode = inputDataNodes.emplace_back();
        dataNode.name = name;
        dataNode.producer = opNodeId;
    }

    mName2DataNode.insert(inputName2DataNode.begin(), inputName2DataNode.end());
    mDataNodes.insert(mDataNodes.end(), inputDataNodes.begin(), inputDataNodes.end());
    mOpNodes.push_back(std::move(opNode));
    return AccDataErrorCode::H_OK;
}

std::string Graph::ToString()
{
    std::ostringstream oss;
    oss << "=== Graph details ===\n";
    for (uint64_t i = 0; i < mOpNodes.size(); ++i) {
        auto &opNode = mOpNodes[i];
        auto &spec = opNode.op->GetSpec();
        oss << "OpNode " << i << ": " << spec.Name() << "\n";
        oss << "\tinputs: ";
        for (auto &input : opNode.inputs) {
            auto &dataNode = mDataNodes[input];
            oss << dataNode.name << "(" << dataNode.producer << "), ";
        }
        oss << "\n\toutputs: ";
        for (auto &output : opNode.outputs) {
            auto &dataNode = mDataNodes[output];
            oss << dataNode.name << ", ";
        }
        oss << "\n";
    }
    return oss.str();
}

AccDataErrorCode Graph::LinkInput(const std::string &name, std::unordered_map<std::string, DataNodeId> &name2DataNode,
                                  std::vector<DataNode> &dataNodes, size_t opNodeId, OpNode &opNode)
{
    auto it = name2DataNode.find(name);
    if (it == name2DataNode.end()) {
        ACCDATA_ERROR("DataNode not exist.");
        return AccDataErrorCode::H_PIPELINE_BUILD_ERROR;
    }
    auto &dataNode = dataNodes[it->second];
    dataNode.consumers.push_back(opNodeId);
    opNode.parents.insert(dataNode.producer);
    opNode.inputs.push_back(it->second);
    auto &parent = mOpNodes[dataNode.producer];
    parent.children.insert(opNodeId);

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Graph::BuildFromPath(const std::vector<OpNodeId> &path)
{
    auto errCode = AccDataErrorCode::H_OK;
    /* Keep the nodes in the path. */
    std::vector<OpNode> opNodes;
    std::vector<DataNode> dataNodes;
    std::unordered_map<std::string, DataNodeId> name2DataNode;
    OpSpec::InOutDesc input;
    OpSpec::InOutDesc output;

    for (auto id : path) {
        auto opNodeId = opNodes.size();
        opNodes.push_back(std::move(mOpNodes[id]));
        auto &opNode = opNodes.back();
        auto &spec = opNode.op->GetSpec();
        /* Associate to Input DataNode and OpNode */
        for (uint64_t i = 0; i < spec.NumInput(); ++i) {
            errCode = spec.GetInput(i, input);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get input.", errCode);

            errCode = LinkInput(input.name, name2DataNode, dataNodes, opNodeId, opNode);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to link input.", errCode);
        }

        std::string inputName;
        for (uint64_t i = 0; i < spec.NumArgInput(); ++i) {
            errCode = spec.GetArgInput(i, inputName);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument input", errCode);

            errCode = LinkInput(inputName, name2DataNode, dataNodes, opNodeId, opNode);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to link input.", errCode);
        }

        /* Add Output DataNode */
        for (uint64_t i = 0; i < spec.NumOutput(); ++i) {
            errCode = spec.GetOutput(i, output);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get output", errCode);
            auto name = output.name;
            if (name2DataNode.count(name) != 0) {
                ACCDATA_ERROR("DataNode already existed.");
                return AccDataErrorCode::H_PIPELINE_BUILD_ERROR;
            }
            DataNodeId dataNodeId = dataNodes.size();
            opNode.outputs.push_back(dataNodeId);
            name2DataNode[name] = dataNodeId;
            auto &dataNode = dataNodes.emplace_back();
            dataNode.name = name;
            dataNode.producer = opNodeId;
        }
    }

    mOpNodes = std::move(opNodes);
    mDataNodes = std::move(dataNodes);
    mName2DataNode = std::move(name2DataNode);

    return errCode;
}

AccDataErrorCode Graph::Build(const std::vector<std::string> &outputs, bool enableFusion)
{
    auto errCode = AccDataErrorCode::H_OK;
    /* Find the path to generate the desired outputs. */
    std::vector<OpNodeId> path;
    errCode = PathToOutputs(outputs, path);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to execute path to outputs.", errCode);

    if (ACCDATA_LIKELY(enableFusion)) {
        errCode = Fuse(outputs, path);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get fuse graph", errCode);
    }

    errCode = BuildFromPath(path);
    if (errCode != AccDataErrorCode::H_OK) {
        return errCode;
    }

    for (auto &output : outputs) {
        auto it = mName2DataNode.find(output);
        if (it == mName2DataNode.end()) {
            ACCDATA_ERROR("output name " << output << " not exists in dataNodes!");
            return AccDataErrorCode::H_PIPELINE_BUILD_ERROR;
        }
        mOutputs.push_back(it->second);
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Graph::PathToOutputs(const std::vector<std::string> &outputs, std::vector<OpNodeId> &path)
{
    for (auto &output : outputs) {
        auto errCode = Traverse(output, path);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to traverse graph", errCode);
    }
    if (path.empty()) {
        ACCDATA_ERROR("No path to generate the outputs");
        return AccDataErrorCode::H_PIPELINE_ERROR;
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Graph::Fuse(const std::vector<std::string> &outputs, std::vector<OpNodeId> &path)
{
    auto errCode = AccDataErrorCode::H_OK;
    std::vector<std::string> fusePlan;
    std::vector<OpNodeId> originPath = path;
    path.clear();

    errCode = FindFusePlan(originPath, outputs, fusePlan);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to find fuse plan", errCode);
    // no validate fusion plan found or no fusion , return origin plan
    if (fusePlan.empty() || fusePlan.size() == originPath.size()) {
        path = originPath;
        return errCode;
    }

    std::vector<OpNode> opNodes;
    int fuseIndex = 0;
    std::vector<OpNodeId> fuseOps;
    for (auto id : originPath) {
        fuseOps.emplace_back(id);
        if (EndWith(fusePlan[fuseIndex], mOpNodes[id].op->GetSpec().Name())) {
            errCode = FuseOps(opNodes, fusePlan[fuseIndex], fuseOps);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to fuse operators", errCode);
            fuseOps.clear();
            fuseIndex++;
        }
    }

    errCode = UpdateGraph(opNodes);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to update graph", errCode);

    errCode = PathToOutputs(outputs, path);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to generate the outputs", errCode);

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Graph::FindFusePlan(const std::vector<OpNodeId> &originPath, const std::vector<std::string> &outputs,
                                     std::vector<std::string> &plan)
{
    std::ostringstream target;
    std::vector<std::string> substrings;
    for (auto id : originPath) {
        target << mOpNodes[id].op->GetSpec().Name();
        substrings.emplace_back(mOpNodes[id].op->GetSpec().Name());
    }
    auto fuseOps = OpFactory::Instance().GetFuseOpsNames();
    std::set<OpNodeId> outputOps;
    for (auto &output : outputs) {
        auto it = mName2DataNode.find(output);
        if (it != mName2DataNode.end()) {
            outputOps.insert(mDataNodes[it->second].producer);
        } else {
            ACCDATA_ERROR("Node name " << output << " not exists!");
        }
    }
    /* find ops with more than 1 output */
    std::set<OpNodeId> branchOps;
    for (auto &id: originPath) {
        if (mOpNodes[id].op->GetSpec().NumOutput() > 1) {
            branchOps.insert(id);
        }
    }
    outputOps.insert(branchOps.begin(), branchOps.end());
    /* output nodes and branch nodes should not fused in the middle */
    for (auto &fuseOp : fuseOps) {
        bool insert = true;
        for (auto id : outputOps) {
            auto pos = fuseOp.find(mOpNodes[id].op->GetSpec().Name());
            if (pos == std::string::npos) {
                continue;
            }
            if (pos + mOpNodes[id].op->GetSpec().Name().size() != fuseOp.size()) {
                insert = false;
                break;
            }
        }
        if (insert) {
            substrings.emplace_back(fuseOp);
        }
    }

    // 查找最小匹配子串集合
    auto fusePlan = FindMinSubStrSet(target.str(), substrings);
    if (fusePlan.empty()) {
        ACCDATA_ERROR("Failed to find fuse plan.");
        return AccDataErrorCode::H_PIPELINE_ERROR;
    }

    if (ValidateFusePlan(fusePlan, originPath, outputs)) {
        plan = fusePlan;
    }

    return AccDataErrorCode::H_OK;
}

bool Graph::ValidateFusePlan(const std::vector<std::string> &plan, const std::vector<OpNodeId> &originPath,
    const std::vector<std::string> &outputs)
{
    /** 校验融合策略：
     * 限制条件1. 输出节点只能是融合算子的最后一个算子，否则不能融合
     *     校验原理：
     *     1）查找融合策略中所有融合算子的最后算子tailNodeIds；
     *     2）查找原pipeline的输出节点outputProducers
     *     3）如果outputProducers不是tailNodeIds子集，则校验不通过，表示某些输出节点被融合丢失了
     */
    int fuseIndex = 0;
    std::set<OpNodeId> tailNodeIds;
    for (auto id : originPath) {
        if (EndWith(plan[fuseIndex], mOpNodes[id].op->GetSpec().Name())) {
            tailNodeIds.insert(id);
            fuseIndex++;
        }
    }
    std::vector<OpNodeId> outputProducers;
    for (auto &output : outputs) {
        outputProducers.emplace_back(mDataNodes[mName2DataNode.find(output)->second].producer);
    }
    for (auto &producer : outputProducers) {
        if (tailNodeIds.find(producer) == tailNodeIds.end()) {
            return false;
        }
    }

    return true;
}

AccDataErrorCode Graph::FuseOps(std::vector<OpNode> &opNodes, const std::string &fuseOpName,
    std::vector<OpNodeId> nodeIds)
{
    auto errCode = AccDataErrorCode::H_OK;
    if (nodeIds.size() == 1) {  // only 1 node, means no fuse on this operation
        opNodes.emplace_back(std::move(mOpNodes[nodeIds[0]]));
        return AccDataErrorCode::H_OK;
    }
    /** step1: create OpSpec for fuseOpName */
    OpSpec fuseSpec(fuseOpName);
    // step1.1: fetch input from first OpNode to be fused
    auto startOpId = nodeIds[0];
    auto numInputs = mOpNodes[startOpId].op->GetSpec().NumInput();
    OpSpec::InOutDesc input;
    for (uint64_t i = 0; i < numInputs; ++i) {
        errCode = mOpNodes[startOpId].op->GetSpec().GetInput(i, input);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get input.", errCode);
        fuseSpec.AddInput(input.name, input.device);
    }
    // step1.2: fetch output from last OpNode to be fused
    auto endOpId = nodeIds[nodeIds.size()-1];
    auto numOutputs = mOpNodes[endOpId].op->GetSpec().NumOutput();
    OpSpec::InOutDesc output;
    for (uint64_t i = 0; i < numOutputs; ++i) {
        errCode = mOpNodes[endOpId].op->GetSpec().GetOutput(i, output);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get output", errCode);
        fuseSpec.AddOutput(output.name, input.device);
    }
    // step1.3: fetch args from all OpNodes to be fused
    std::string inputName;
    for (auto &i : nodeIds) {
        auto spec = mOpNodes[i].op->GetSpec();
        for (auto &argInput : spec.GetArgInputIdxs()) {
            errCode = spec.GetArgInput(argInput.second, inputName);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument input.", errCode);
            fuseSpec.AddArgInput(argInput.first, inputName);
        }
        for (auto &arg : spec.GetArgIdxs()) {
            auto opArg = spec.GetOpArg(arg.second, errCode);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get operator argument.",
                                           errCode);
            fuseSpec.AddArg(arg.first, opArg);
        }
    }
    /** step2: add fused opnode to opNodes */
    auto &opNode = opNodes.emplace_back();
    errCode = OpFactory::Instance().Create(fuseSpec.Name(), fuseSpec, opNode.op);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to create op factory instance.", errCode);
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Graph::UpdateGraph(std::vector<OpNode> &opNodes)
{
    // clear all state and start adding opnode again
    mOpNodes = std::move(opNodes);
    mDataNodes.clear();
    mName2DataNode.clear();

    for (size_t nodeId = 0; nodeId < mOpNodes.size(); ++nodeId) {
        auto &spec = mOpNodes[nodeId].op->GetSpec();
        OpSpec::InOutDesc output;
        for (uint64_t i = 0; i < spec.NumOutput(); ++i) {
            auto errCode = spec.GetOutput(i, output);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get output", errCode);
            auto name = output.name;
            if (mName2DataNode.count(name) != 0) {
                ACCDATA_ERROR("The DataNode already existed.");
                return AccDataErrorCode::H_PIPELINE_ERROR;
            }
            mName2DataNode[name] = mDataNodes.size();
            auto &dataNode = mDataNodes.emplace_back();
            dataNode.name = name;
            dataNode.producer = nodeId;
        }
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Graph::Traverse(const std::string &output, std::vector<OpNodeId> &path)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto it = mName2DataNode.find(output);
    if (it == mName2DataNode.end()) {
        ACCDATA_ERROR("No datanode found for output.");
        return AccDataErrorCode::H_PIPELINE_ERROR;
    }

    auto &dataNode = mDataNodes[it->second];
    auto opNodeId = dataNode.producer;
    if (std::find(path.begin(), path.end(), opNodeId) != path.end()) {
        return AccDataErrorCode::H_OK;
    }

    auto &opNode = mOpNodes[dataNode.producer];
    auto &spec = opNode.op->GetSpec();
    OpSpec::InOutDesc input;
    std::string name;
    for (uint64_t i = 0; i < spec.NumInput(); ++i) {
        errCode = spec.GetInput(i, input);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get input", errCode);
        Traverse(input.name, path);
    }
    for (uint64_t i = 0; i < spec.NumArgInput(); ++i) {
        errCode = spec.GetArgInput(i, name);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument input", errCode);
        Traverse(name, path);
    }
    path.push_back(opNodeId);
    return AccDataErrorCode::H_OK;
}

} // namespace accdata
} // namespace acclib
