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
 * @Date: 2025-2-20 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-20 9:00:00
 */
#include <gtest/gtest.h>

#include "pipeline/graph/graph.h"

namespace {
using namespace acclib::accdata;

class TestPipelineGraph : public ::testing::Test {
public:
    void SetUp()
    {
    }
    void TearDown()
    {
    }
};


TEST_F(TestPipelineGraph, AddGetNodeSuccess)
{
    Graph graph;
    AccDataErrorCode errCode = AccDataErrorCode::H_OK;

    OpSpec resizeCrop("ResizeCrop");
    resizeCrop.AddOutput("ResizeOutput1", "cpu");
    errCode = graph.AddNode(resizeCrop);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(graph.NumOpNode(), 1);

    OpSpec resizeCrop2("ResizeCrop");
    errCode = graph.AddNode(resizeCrop2);
    resizeCrop.AddOutput("ResizeOutput2", "cpu");
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(graph.NumOpNode(), 2); // Add 2 opNode

    auto& opNode = graph.GetOpNode(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    auto& opNodeError = graph.GetOpNode(3, errCode); // 3 is error opNodeId
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);

    DataNodeId nodeId;
    errCode = graph.GetDataNodeId("ResizeOutput1", nodeId);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    auto& opDataNode = graph.GetDataNode(nodeId, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = graph.GetDataNodeId("ResizeOutput3", nodeId);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);

    auto& opDataNodeError = graph.GetDataNode(3, errCode); // 3 is error opDataNode
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineGraph, AddNodeNotExist)
{
    Graph graph;
    AccDataErrorCode errCode = AccDataErrorCode::H_OK;

    OpSpec ErrorNode("ErrorNode");
    errCode = graph.AddNode(ErrorNode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    EXPECT_EQ(graph.NumOpNode(), 0);
}

}