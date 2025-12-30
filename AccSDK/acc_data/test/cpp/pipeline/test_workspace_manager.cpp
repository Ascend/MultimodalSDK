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
 * @Date: 2025-4-2 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-4-2 9:00:00
 */
#include <gtest/gtest.h>

#include "pipeline/workspace/workspace_manager.h"
#include "operator/op_spec.h"

namespace {

using namespace acclib::accdata;

class TestWorkspaceManager : public ::testing::Test {
public:
    void SetUp()
    {
        buffer.str(std::string());  // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }
    void TearDown()
    {
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;
    }

    void InitMgr(WorkspaceManager& mgr)
    {
        OpSpec externalInput("ExternalSource");
        externalInput.AddOutput("ExternalSource", "cpu");
        OpSpec totensor("ToTensor");
        totensor.AddInput("ExternalSource", "cpu");
        totensor.AddArg("layout", static_cast<int64_t>(TensorLayout::NCHW), false);
        totensor.AddOutput("ToTensor", "cpu");

        graph.AddNode(externalInput);
        graph.AddNode(totensor);
    }

    WorkspaceManager mgr;
    Graph graph;
    std::vector<int> queueDepth = {2, 2}; // 2 is min
    int maxBatchSize = 2; // maxBatchSize is 2
    std::stringstream buffer;
    std::streambuf *sbuf;
};

TEST_F(TestWorkspaceManager, TestInitSuccess)
{
    InitMgr(mgr);
    EXPECT_EQ(mgr.Init(graph, queueDepth, maxBatchSize, nullptr), AccDataErrorCode::H_OK);
}

TEST_F(TestWorkspaceManager, TestInitFailed)
{
    InitMgr(mgr);
    auto errCode = mgr.Init(graph, { 1 }, maxBatchSize, nullptr);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestWorkspaceManager, TestAcquireRelease)
{
    InitMgr(mgr);
    mgr.Init(graph, queueDepth, maxBatchSize, nullptr);
    auto idx = mgr.Acquire();
    auto ws = mgr.GetWorkspace(idx, 0);
    auto dataNode = mgr.GetDataStore(idx, 0);
    mgr.Release(idx);
    mgr.AcquireOutputIdx();
    mgr.ReleaseOutputIdx();
}

TEST_F(TestWorkspaceManager, TestNumberDataNodeNotEqualQueueDepth) // Graph中的数据节点数不等于队列深度的大小，WorkspaceManger初始化失败
{
    queueDepth = {2};
    InitMgr(mgr);
    EXPECT_EQ(mgr.Init(graph, queueDepth, maxBatchSize, nullptr), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestWorkspaceManager, TestQueueDepthLessThanOne) // 队列深度中的值小于等于0，WorkspaceManger初始化失败
{
    queueDepth = {0, 0};
    InitMgr(mgr);
    EXPECT_EQ(mgr.Init(graph, queueDepth, maxBatchSize, nullptr), AccDataErrorCode::H_PIPELINE_ERROR);
}

}
