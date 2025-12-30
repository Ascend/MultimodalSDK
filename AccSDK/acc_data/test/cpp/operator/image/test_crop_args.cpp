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

#include "operator/image/crop_args.h"
#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"
#include "interface/accdata_error_code.h"

namespace {
using namespace acclib::accdata;

class TestCropArgs : public ::testing::Test {
public:
    void SetUp()
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<float>("crop_pos_x", 1.0f);
        opSpec->AddArg<float>("crop_pos_y", 1.0f);
        opSpec->AddArg<std::string>("round_mode", "truncate");
        opSpec->AddArg<std::vector<int64_t>>("crop", {1080LL, 1920LL});

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

TEST_F(TestCropArgs, SetUpSuccessRoundModeTruncate)
{
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestCropArgs, SetUpSuccessRoundModeRound)
{
    opSpec->AddArg<std::string>("round_mode", "round");
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestCropArgs, SetUpSuccessRoundModeDefault)
{
    OpSpec* tmpOpSpec = new OpSpec("testOpSpec1");
    tmpOpSpec->AddArg<float>("crop_pos_x", 1.0f);
    tmpOpSpec->AddArg<float>("crop_pos_y", 1.0f);
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", {1080LL, 1920LL});
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*tmpOpSpec, *workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestCropArgs, SetUpCropError)
{
    opSpec->AddArg<std::vector<int64_t>>("crop", {1LL});
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, SetUpRoundModeError)
{
    opSpec->AddArg<std::string>("round_mode", "Invalid");
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, SetUpCropPosXLargerThanOneError)
{
    opSpec->AddArg<float>("crop_pos_x", 1.5f);
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, SetUpCropPosXErrorLessThanZeroError)
{
    opSpec->AddArg<float>("crop_pos_x", -0.5f);
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, SetUpCropPosYLargerThanOneError)
{
    opSpec->AddArg<float>("crop_pos_y", 1.5f);
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, SetUpCropPosYLessThanZeroError)
{
    opSpec->AddArg<float>("crop_pos_y", -0.5f);
    CropArgs cropArgs;
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, GetCropTop)
{
    CropArgs cropArgs;
    cropArgs.Setup(*opSpec, *workspace);
    int64_t height = 1080LL; // height与cropArgs的mHeight一样大
    EXPECT_EQ(cropArgs.Top(height), 0);

    height = 1090LL; // height比cropArgs的mHeight大
    EXPECT_EQ(cropArgs.Top(height), 1 * 10LL); // cropPosX * diff

    height = 1070LL; // height比cropArgs的mHeight小
    EXPECT_EQ(cropArgs.Top(height), 0);
}

TEST_F(TestCropArgs, GetCropLeft)
{
    CropArgs cropArgs;
    cropArgs.Setup(*opSpec, *workspace);
    int64_t width = 1920LL; // width与cropArgs的mWidth一样大
    EXPECT_EQ(cropArgs.Left(width), 0);

    width = 1930LL; // width比cropArgs的mWidth大
    EXPECT_EQ(cropArgs.Left(width), 1 * 10LL); // cropPosY * diff

    width = 1910LL; // width比cropArgs的mWidth小
    EXPECT_EQ(cropArgs.Left(width), 0);
}

TEST_F(TestCropArgs, GetWidth)
{
    CropArgs cropArgs;
    cropArgs.Setup(*opSpec, *workspace);
    EXPECT_EQ(cropArgs.Width(), 1920LL);
}

TEST_F(TestCropArgs, GetHeight)
{
    CropArgs cropArgs;
    cropArgs.Setup(*opSpec, *workspace);
    EXPECT_EQ(cropArgs.Height(), 1080LL);
}

TEST_F(TestCropArgs, HeightValueLessThan10)
{
    CropArgs cropArgs;
    opSpec->AddArg<std::vector<int64_t>>("crop", {5LL, 1024LL});
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, HeightValueLargerThan8192)
{
    CropArgs cropArgs;
    opSpec->AddArg<std::vector<int64_t>>("crop", {8193LL, 1024LL});
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, WidthValueLessThan10)
{
    CropArgs cropArgs;
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 5LL});
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestCropArgs, WidthValueLargerThan8192)
{
    CropArgs cropArgs;
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 8193LL});
    EXPECT_EQ(cropArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}
}