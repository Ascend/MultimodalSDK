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
 * Description:
 * Author: Dev
 * Create: 2025-03-29
 */

#include <gtest/gtest.h>

#include "operator/math/normalize_args.h"

namespace {
using namespace acclib::accdata;

class TestNormalizeArgs : public ::testing::Test {
public:
    void PrepareOpSpec()
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<std::vector<float>>("mean", {1.0f, 1.0f, 1.0f});
        opSpec->AddArg<std::vector<float>>("stddev", {1.0f, 1.0f, 1.0f});
    }

    void PrepareWorkSpace()
    {
        workspace = new Workspace();
    }

    void SetUp()
    {
        PrepareOpSpec();
        PrepareWorkSpace();

        buffer.str(std::string());  // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }

    void TearDown()
    {
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;

        OpSpec* opSpecDeleter = opSpec;
        delete opSpecDeleter;
        Workspace* workspaceDeleter = workspace;
        delete workspaceDeleter;
        opSpec = nullptr;
        workspace = nullptr;
    }

    OpSpec *opSpec = nullptr;
    Workspace *workspace = nullptr;
    std::stringstream buffer;
    std::streambuf *sbuf;
};

TEST_F(TestNormalizeArgs, TestSetupNotScaleSuccess) // 运行成功无报错
{
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestNormalizeArgs, TestSetupWithScaleSuccess) // 运行成功无报错
{
    opSpec->AddArg<float>("scale", 1.0f);
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestNormalizeArgs, TestSetupGetMeanFailed) // mean数据类型不合法，运行失败
{
    opSpec->AddArg<std::vector<int64_t>>("mean", {1, 1, 1});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Failed to get the mean argument."), std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupGetStddevFailed) // stddev数据类型不合法，运行失败
{
    opSpec->AddArg<std::vector<int64_t>>("stddev", {1, 1, 1});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Failed to get the stddev argument."), std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupMeanValueError) // mean不在[0, 1]范围之内
{
    opSpec->AddArg<std::vector<float>>("mean", {0.458f, 1.1f});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalizeArgs, TestSetupMeanSizeError) // channel=3，mean维度与channel数不一致，运行失败
{
    opSpec->AddArg<std::vector<float>>("mean", {0.458f, 0.458f});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("The mean argument and the stddev argument should have 3 elements."),
              std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupStddevSizeError) // channel=3，stddev维度与channel数不一致，运行失败
{
    opSpec->AddArg<std::vector<float>>("stddev", {0.229f, 0.229f});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("The mean argument and the stddev argument should have 3 elements."),
              std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupWithInvalidScale) // scale数据类型不合法，运行失败
{
    opSpec->AddArg<int64_t>("scale", 1);
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Failed to get the scale argument."), std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupWithStddevZero)
{
    opSpec->AddArg<std::vector<float>>("stddev", {0.0f, 0.0f, 0.0f});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("stddev must be larger than zero."), std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupWithNegativeStddev)
{
    opSpec->AddArg<std::vector<float>>("stddev", {-0.8f, -0.8f, -0.8f});
    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("stddev must be larger than zero."), std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupWithoutStddev)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("mean", {0.354f, 0.354f, 0.354f});
    operatorSpec->AddArg<float>("scale", 1.0);

    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*operatorSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Failed to get the stddev argument."), std::string::npos);
}

TEST_F(TestNormalizeArgs, TestSetupWithoutMean)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("stddev", {0.501f, 0.501f, 0.501f});
    operatorSpec->AddArg<float>("scale", 1.0);

    NormalizeArgs normalizeArgs;
    EXPECT_EQ(normalizeArgs.Setup(*operatorSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Failed to get the mean argument."), std::string::npos);
}

}