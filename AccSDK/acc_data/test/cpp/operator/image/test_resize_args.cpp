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
 * Create: 2025-03-25
 */
#include <gtest/gtest.h>

#include "operator/image/resize_args.h"
#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"
#include "interface/accdata_error_code.h"

namespace {
    using namespace acclib::accdata;

    class TestResizeArgs : public ::testing::Test {
    public:
        void SetUp()
        {
            opSpec = new OpSpec("testOpSpec");
            opSpec->AddArg<std::string>("interpolation_mode", "bilinear");
            opSpec->AddArg<std::vector<int64_t>>("resize", {1080LL, 1920LL});

            Logger::SetLogLevelStr("debug");  // capture debug info
            buffer.str(std::string());  // clears the buffer.
            sbuf = std::cout.rdbuf();
            std::cout.rdbuf(buffer.rdbuf());
        }
        void TearDown()
        {
            OpSpec* opSpecDeleter = opSpec;
            delete opSpecDeleter;
            Workspace* workspaceDeleter = workspace;
            delete workspaceDeleter;
            opSpec = nullptr;
            workspace = nullptr;

            Logger::SetLogLevelStr("info");
            std::cout.rdbuf(sbuf);
            std::cout << buffer.str() << std::endl;
        }

        OpSpec* opSpec = nullptr;
        Workspace* workspace = nullptr;
        std::stringstream buffer;
        std::streambuf *sbuf;
    };

    TEST_F(TestResizeArgs, SetUpSuccessBilinear)
    {
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
    }

    TEST_F(TestResizeArgs, SetUpSuccessBicubic)
    {
        opSpec->AddArg<std::string>("interpolation_mode", "bicubic");
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
    }

    TEST_F(TestResizeArgs, SetUpSuccessDefault)
    {
        OpSpec* tmpOpSpec = new OpSpec("testOpSpec1");
        tmpOpSpec->AddArg<std::vector<int64_t>>("resize", {1080LL, 1920LL});
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*tmpOpSpec, *workspace), AccDataErrorCode::H_OK);
        delete tmpOpSpec;
        tmpOpSpec = nullptr;
    }

    TEST_F(TestResizeArgs, SetUpResizeError)
    {
        opSpec->AddArg<std::vector<int64_t>>("resize", {1080LL});
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    }

    TEST_F(TestResizeArgs, SetUpInvalidInterpolationMode)
    {
        opSpec->AddArg<std::string>("interpolation_mode", "Invalid");
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    }

    TEST_F(TestResizeArgs, HeightValueLessThan10)
    {
        opSpec->AddArg<std::vector<int64_t>>("resize", {5LL, 1920LL});
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    }

    TEST_F(TestResizeArgs, HeightValueLargerThan8192)
    {
        opSpec->AddArg <std::vector<int64_t>>("resize", {8193LL, 1920LL});
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    }

    TEST_F(TestResizeArgs, WidthValueLessThan10)
    {
        opSpec->AddArg<std::vector<int64_t>>("resize", {1080LL, 5LL});
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    }

    TEST_F(TestResizeArgs, WidthValueLargerThan8192)
    {
        opSpec->AddArg<std::vector<int64_t>>("resize", {1080LL, 8193LL});
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
    }

        TEST_F(TestResizeArgs, GetParameterTest)
    {
        ResizeArgs args;
        EXPECT_EQ(args.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
        EXPECT_EQ(args.Height(), 1080LL);
        EXPECT_EQ(args.Width(), 1920LL);
        EXPECT_EQ(args.Mode(), InterpMode::BILINEAR);
    }
}
