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
 * @Date: 2025-3-29 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-29 17:00:00
 */

#include <gtest/gtest.h>

#include "operator/op_spec.h"
#include "interface/accdata_op_spec.h"

namespace {
using namespace acclib::accdata;

TEST(TestAccDataOpSpec, CreateOpSpecFailed)
{
    std::string opName = "a";
    for (uint64_t i = 0; i < 1024ULL; ++i) {
        opName.append("a");
    }
    EXPECT_EQ(AccDataOpSpec::Create(opName), nullptr);
}

TEST(TestAccDataOpSpec, AddInputFailed)
{
    std::string opName = "Norm";
    auto opSpec = AccDataOpSpec::Create(opName);
    std::string successName = "data";
    opSpec->AddInput(successName, "cpu");
    std::string failName = "a";
    for (uint64_t i = 0; i < 1024ULL; ++i) {
        failName.append("a");
    }
    opSpec->AddInput(failName, "cpu");
}

TEST(TestAccDataOpSpec, AddOutputFailed)
{
    std::string opName = "Norm";
    auto opSpec = AccDataOpSpec::Create(opName);
    std::string successName = "data";
    opSpec->AddInput(successName, "cpu");
    std::string failName = "a";
    for (uint64_t i = 0; i < 1024ULL; ++i) {
        failName.append("a");
    }
    opSpec->AddOutput(failName, "cpu");
}

TEST(TestAccDataOpSpec, AddArgFailed)
{
    std::string opName = "Norm";
    auto opSpec = AccDataOpSpec::Create(opName);
    std::string name = "a";
    for (uint64_t i = 0; i < 1024ULL; ++i) {
        name.append("a");
    }
    float value = 1.0F;
    opSpec->AddArg(name, value);
}

class TestOpSpec : public ::testing::Test {
public:
    void SetUp()
    {
        opSpec = new OpSpec("Normalize");
        opSpec->AddInput("image", "cpu");
        opSpec->AddArgInput("scale", "rand");
        opSpec->AddArg<float>("scale", 0.5f);
        opSpec->AddOutput("normalize", "cpu");
    }

    void TearDown()
    {
        OpSpec* opSpecDeleter = opSpec;
        delete opSpecDeleter;
        opSpec = nullptr;
    }

    OpSpec *opSpec;
    std::string argInput = "";
    OpSpec::InOutDesc input;
    OpSpec::InOutDesc output;
    AccDataErrorCode errCode = AccDataErrorCode::H_OK;
    float value = 0.0;
};

TEST_F(TestOpSpec, AddInputSuccess)
{
    EXPECT_EQ(opSpec->NumInput(), 1);
}

TEST_F(TestOpSpec, AddOutputSuccess)
{
    EXPECT_EQ(opSpec->NumOutput(), 1);
}

TEST_F(TestOpSpec, AddNewArgInputSuccess)
{
    EXPECT_EQ(opSpec->NumArgInput(), 1);
}

TEST_F(TestOpSpec, AddRepeatArgInputOverwriteSuccess)
{
    opSpec->AddArgInput("scale", "default");
    opSpec->GetArgInput(0, argInput);
    EXPECT_EQ(argInput, "default");
}

TEST_F(TestOpSpec, AddRepeatArgInputNotOverwriteSuccess)
{
    opSpec->AddArgInput("scale", "default", false);
    opSpec->GetArgInput(0, argInput);
    EXPECT_EQ(argInput, "rand");
}

TEST_F(TestOpSpec, AddNewArgSuccess)
{
    EXPECT_EQ(opSpec->NumArg(), 1);
}

TEST_F(TestOpSpec, AddRepeatArgOverwriteSuccess)
{
    opSpec->AddArg<float>("scale", 0.6f);
    auto opArg = opSpec->GetOpArg(0, errCode);
    opArg->Value(value);
    EXPECT_EQ(value, 0.6f);
}

TEST_F(TestOpSpec, AddRepeatArgNotOverwriteSuccess)
{
    opSpec->AddArg<float>("scale", 0.6f, false);
    auto opArg = opSpec->GetOpArg(0, errCode);
    opArg->Value(value);
    EXPECT_EQ(value, 0.5f);
}

TEST_F(TestOpSpec, AddNewArgWithValueIsOpArgSuccess) // 算子变量的value类型为OpArg，添加该算子变量成功
{
    int numArg = 2;
    auto opArgMean = OpArg::Create("mean", 1.0f);
    opSpec->AddArg("mean", opArgMean);
    EXPECT_EQ(opSpec->NumArg(), numArg);
}

TEST_F(TestOpSpec, AddRepeatArgWithValueIsOpArgOverwriteSuccess) // 算子变量的value类型为OpArg，该变量先前存在，重写变量，添加成功
{
    auto opArgScale = OpArg::Create("scale", 1.0f);
    opSpec->AddArg("scale", opArgScale);
    auto opArg = opSpec->GetOpArg(0, errCode);
    opArg->Value(value);
    EXPECT_EQ(value, 1.0f);
}

TEST_F(TestOpSpec, AddRepeatArgWithValueIsOpArgNotOverwriteSuccess) // 算子变量的value类型为OpArg，该变量先前存在，不重写变量
{
    auto opArgScale = OpArg::Create("scale", 1.0f);
    opSpec->AddArg("scale", opArgScale, false);
    auto opArg = opSpec->GetOpArg(0, errCode);
    opArg->Value(value);
    EXPECT_EQ(value, 0.5f);
}

TEST_F(TestOpSpec, GetInputSuccess)
{
    EXPECT_EQ(opSpec->GetInput(0, input), AccDataErrorCode::H_OK);
}

TEST_F(TestOpSpec, GetInputFailed)
{
    EXPECT_EQ(opSpec->GetInput(1, input), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpSpec, GetOutputSuccess)
{
    EXPECT_EQ(opSpec->GetOutput(0, output), AccDataErrorCode::H_OK);
}

TEST_F(TestOpSpec, GetOutputFailed)
{
    EXPECT_EQ(opSpec->GetOutput(1, input), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpSpec, GetArgInputSuccess)
{
    EXPECT_EQ(opSpec->GetArgInput(0, argInput), AccDataErrorCode::H_OK);
}

TEST_F(TestOpSpec, GetArgInputFailed)
{
    EXPECT_EQ(opSpec->GetArgInput(1, argInput), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpSpec, GetOpArgSuccess)
{
    opSpec->GetOpArg(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestOpSpec, GetOpArgFailed)
{
    opSpec->GetOpArg(1, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpSpec, FromNameGetArgInputSuccess)
{
    EXPECT_EQ(opSpec->GetArgInput("scale", argInput), AccDataErrorCode::H_OK);
}

TEST_F(TestOpSpec, FromNameGetArgInputFailed)
{
    EXPECT_EQ(opSpec->GetArgInput("mean", argInput), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpSpec, TryGetArgInputSuccess)
{
    EXPECT_TRUE(opSpec->TryGetArgInput("scale", argInput));
}

TEST_F(TestOpSpec, TryGetArgInputFailed)
{
    EXPECT_FALSE(opSpec->TryGetArgInput("mean", argInput));
}

TEST_F(TestOpSpec, GetArgSuccess)
{
    EXPECT_EQ(opSpec->GetArg("scale", value), AccDataErrorCode::H_OK);
}

TEST_F(TestOpSpec, GetArgFailed)
{
    EXPECT_EQ(opSpec->GetArg("mean", value), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpSpec, TryGetArgSuccess)
{
    EXPECT_TRUE(opSpec->TryGetArg("scale", value));
}

TEST_F(TestOpSpec, TryGetArgFailed) // 变量不存在，获取失败，返回false
{
    EXPECT_FALSE(opSpec->TryGetArg("mean", value));
}

TEST_F(TestOpSpec, TryGetArgWithInvalidTypeFailed) // 传入value类型与OpSpec中的变量类型不一致，获取失败，返回false
{
    int intValue = 0;
    EXPECT_FALSE(opSpec->TryGetArg("scale", intValue));
}

class TestFromOpSpecOrWorkspaceGetArg : public ::testing::Test {
public:
    void SetUp()
    {
        opSpec = new OpSpec("Normalize");
        opSpec->AddArg<float>("scale", 0.5f);
        opSpec->AddArgInput("mean", "rand");
        input = std::make_shared<TensorList>(1);
    }

    void TearDown()
    {
        OpSpec* opSpecDeleter = opSpec;
        delete opSpecDeleter;
        opSpec = nullptr;
    }

    OpSpec *opSpec;
    Workspace ws;
    float floatValue = 0.0;
    std::string stringValue = " ";
    std::shared_ptr<TensorList> input;
    TensorShape tensorShape{1, 3, 1, 1};
    TensorShape OneTensorShape{1, 1, 1, 1};
};

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromOpSpecGetArgSuccess)
{
    EXPECT_EQ(opSpec->GetArg("scale", ws, floatValue), AccDataErrorCode::H_OK);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromOpSpecGetArgFailed)
{
    EXPECT_EQ(opSpec->GetArg("stddev", ws, floatValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetNotFoundVectorArg)  // Get Failed
{
    std::vector<float> vectorValue;
    EXPECT_EQ(opSpec->GetArg("mean", ws, vectorValue), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetEmptyTensorVectorArg)  // Get Failed
{
    std::vector<float> vectorValue;
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, vectorValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetCharDatatypeVectorArgSuccess)
{
    std::vector<std::string> vectorValue;
    std::vector<char> data = {'r', 'r', 'r'};
    input->operator[](0).Copy<char>(data.data(), tensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, vectorValue), AccDataErrorCode::H_OK);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetCharDatatypeVectorArgFailed)
{
    std::vector<std::string> vectorValue;
    std::vector<float> data = {0.5f, 0.5f, 0.5f};
    input->operator[](0).Copy<float>(data.data(), tensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, vectorValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetValidDatatypeVectorArg)  // Get Success
{
    std::vector<float> vectorValue;
    std::vector<float> data = {0.5f, 0.5f, 0.5f};
    input->operator[](0).Copy<float>(data.data(), tensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, vectorValue), AccDataErrorCode::H_OK);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetInvalidDatatypeVectorArg)  // Get Failed
{
    std::vector<float> vectorValue;
    std::vector<char> data = {'r', 'r', 'r'};
    input->operator[](0).Copy<char>(data.data(), tensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, vectorValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetNotFoundOneArg)  // Get Failed
{
    EXPECT_EQ(opSpec->GetArg("mean", ws, floatValue), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetEmptyTensorListOneArg)  // Get Failed
{
    auto emptyTensorList = std::make_shared<TensorList>(0);
    ws.AddArgInput("mean", emptyTensorList);
    EXPECT_EQ(opSpec->GetArg("mean", ws, floatValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetEmptyTensorOneArg)  // Get Failed
{
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, floatValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetCharDatatypeOneArgSuccess)
{
    std::vector<char> data = {'r'};
    input->operator[](0).Copy<char>(data.data(), OneTensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, stringValue), AccDataErrorCode::H_OK);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetCharDatatypeOneArgFailed)
{
    std::vector<float> data = {0.5f};
    input->operator[](0).Copy<float>(data.data(), OneTensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, stringValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetValidDatatypeOneArg)  // Get Success
{
    std::vector<float> data = {0.5f};
    input->operator[](0).Copy<float>(data.data(), OneTensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, floatValue), AccDataErrorCode::H_OK);
}

TEST_F(TestFromOpSpecOrWorkspaceGetArg, FromWorkspaceGetInvalidDatatypeOneArg)  // Get Failed
{
    std::vector<char> data = {'r'};
    input->operator[](0).Copy<char>(data.data(), OneTensorShape);
    ws.AddArgInput("mean", input);
    EXPECT_EQ(opSpec->GetArg("mean", ws, floatValue), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

}