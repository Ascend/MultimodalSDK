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
 * @Date: 2025-3-28 10:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-28 10:00:00
 */

#include <utility>
#include <random>

#include <gtest/gtest.h>

#include "operator/fusion/to_tensor_resize_crop_normalize.h"

namespace {
using namespace acclib::accdata;

constexpr int PARAM_INTER_MODE = 0;
constexpr int PARAM_ROUND_MODE = 1;
constexpr int PARAM_SCALE = 2;
constexpr int PARAM_ORIGIN_SIZE = 3;
constexpr int PARAM_RESIZED_SIZE = 4;
constexpr int PARAM_CROPPED_SIZE = 5;
constexpr int PIXEL = 255;

class BaseTestFusedOp {
protected:
    template<typename T>
    void GenerateTensorDatas(size_t size, std::vector<T> &datas)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, PIXEL);

        for (int i = 0; i < size; ++i) {
            datas[i] = dis(gen);
        }
    }

    void PrepareWorkSpace()
    {
        size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
        std::vector<uint8_t> datas(tensorSize);
        GenerateTensorDatas<uint8_t>(tensorSize, datas);

        auto mInputTensor = std::make_shared<TensorList>(1);
        TensorShape tensorShape = {1, originImageSize.first, originImageSize.second, 3};
        mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
        mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
        auto mOutputTensor = std::make_shared<TensorList>(1);
        auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

        workspace = new Workspace();
        workspace->SetThreadPool(mThreadPool);
        workspace->AddInput(mInputTensor);
        workspace->AddOutput(mOutputTensor);
    }

    void PrepareOpSpec(bool withOutput = true)
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<std::vector<int64_t>>("resize", {resize.first, resize.second});
        opSpec->AddArg<std::vector<int64_t>>("crop", {crop.first, crop.second});
        opSpec->AddArg<std::string>("interpolation_mode", interpolationMode);
        opSpec->AddArg<std::string>("round_mode", roundMode);
        opSpec->AddArg<float>("crop_pos_x", 0.5f);
        opSpec->AddArg<float>("crop_pos_y", 0.5f);
        opSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f, 0.5f});
        opSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f, 0.5f});
        opSpec->AddArg<float>("scale", scale);
        opSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NCHW));
        if (withOutput) {
            opSpec->AddOutput("testOutput", "cpu");
        }
    }

    OpSpec* opSpec = nullptr;
    Workspace* workspace = nullptr;
    std::string interpolationMode {"bilinear"};
    std::string roundMode {"round"};
    float scale {1.0f};
    std::pair<size_t, size_t> originImageSize = std::make_pair(1080LL, 1920LL);
    std::pair<int64_t, int64_t> resize = std::make_pair(1080LL, 1920LL);
    std::pair<int64_t, int64_t> crop = std::make_pair(1024LL, 1024LL);
};

class TestFusedOp : public ::testing::Test, public BaseTestFusedOp {
public:
    void SetUp()
    {
        PrepareWorkSpace();

        Logger::SetLogLevelStr("debug");  // capture debug info
        buffer.str(std::string());  // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }

    void TearDown()
    {
        workspace->Clear();
        Logger::SetLogLevelStr("info");
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;

        OpSpec* opSpecDeleter = opSpec;
        delete opSpecDeleter;
        Workspace* workspaceDeleter = workspace;
        delete workspaceDeleter;
        opSpec = nullptr;
        workspace = nullptr;
    }

    std::stringstream buffer;
    std::streambuf *sbuf;
};

class ParamTestFusedOp : public ::testing::TestWithParam<std::tuple<std::string, std::string, float,
                                                                    std::pair<int64_t, int64_t>,
                                                                    std::pair<float, float>,
                                                                    std::pair<float, float>>>,
                         public BaseTestFusedOp {
public:
    void SetUp()
    {
        interpolationMode = std::get<PARAM_INTER_MODE>(GetParam());
        roundMode = std::get<PARAM_ROUND_MODE>(GetParam());
        scale = std::get<PARAM_SCALE>(GetParam());
        originImageSize = std::get<PARAM_ORIGIN_SIZE>(GetParam());
        resize = std::get<PARAM_RESIZED_SIZE>(GetParam());
        crop = std::get<PARAM_CROPPED_SIZE>(GetParam());
        PrepareOpSpec();
        PrepareWorkSpace();
    }

    void TearDown()
    {
        workspace->Clear();
        OpSpec* opSpecDeleter = opSpec;
        delete opSpecDeleter;
        Workspace* workspaceDeleter = workspace;
        delete workspaceDeleter;
        opSpec = nullptr;
        workspace = nullptr;
    }
};

INSTANTIATE_TEST_SUITE_P(
    TestFusedOpCases,
    ParamTestFusedOp,
    ::testing::Values(
        // interpolation_mode round_mode scale originSize resize crop
        std::make_tuple("bilinear", "truncate", 1.0f, std::make_pair(1080LL, 1920LL),
                        std::make_pair(1080LL, 1920LL), std::make_pair(1024LL, 1024LL)),
        std::make_tuple("bilinear", "round", 1.0f, std::make_pair(1080LL, 1920LL),
                        std::make_pair(1080LL, 1920LL), std::make_pair(1024LL, 1024LL)),
        std::make_tuple("bilinear", "truncate", 0.5f, std::make_pair(1080LL, 1920LL),
                        std::make_pair(1080LL, 1920LL), std::make_pair(1024LL, 1024LL)),
        std::make_tuple("bilinear", "round", 0.5f, std::make_pair(1080LL, 1920LL),
                        std::make_pair(1080LL, 1920LL), std::make_pair(1024LL, 1024LL))
        ),
    [](const testing::TestParamInfo<ParamTestFusedOp::ParamType>& info) {
        std::string interpolation_mode = std::get<PARAM_INTER_MODE>(info.param);
        std::string round_mode = std::get<PARAM_ROUND_MODE>(info.param);
        float scale = std::get<PARAM_SCALE>(info.param);
        auto str_scale = (scale == 1.0 ? "" : "scaled");
        return "interpolation_mode_" + interpolation_mode + "round_mode_" + round_mode + str_scale;
    }
);

TEST_P(ParamTestFusedOp, TestRunSuccess)
{
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NCHW);
}

TEST_F(TestFusedOp, TestRunWithWrongTensorList)
{
    PrepareOpSpec();
    workspace->Clear();
    auto mInputTensor = std::make_shared<TensorList>();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithInconsistentOutputs)
{
    PrepareOpSpec();
    workspace->Clear();
    auto mInputTensor = std::make_shared<TensorList>();
    workspace->AddInput(mInputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithoutInputs)
{
    PrepareOpSpec();
    workspace->Clear();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace->AddOutput(mOutputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestFusedOp, TestRunWithoutOutputs)
{
    PrepareOpSpec(false);
    workspace->Clear();

    size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, originImageSize.first, originImageSize.second, 3};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestFusedOp, TestRunWithUnsupportedDataType)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
    std::vector<float> datas(tensorSize);
    GenerateTensorDatas<float>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, originImageSize.first, originImageSize.second, 3};
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithUnsupportedDataLayout)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 3, originImageSize.first, originImageSize.second};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestFusedOp, TestRunWithUnsupportedNCHWToNHWC)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NHWC));
    workspace->Clear();
    size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 3, originImageSize.first, originImageSize.second};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestFusedOp, TestRunWithInvalidDataLayout)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 3, originImageSize.first, originImageSize.second};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::LAST);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithUnsupportedInterpolation)
{
    PrepareOpSpec();
    opSpec->AddArg<std::string>("interpolation_mode", "bicubic");
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_FUSIONOP_ERROR);
}

TEST_F(TestFusedOp, TestRunWithInvalidInterpolation)
{
    PrepareOpSpec();
    opSpec->AddArg<std::string>("interpolation_mode", "invalid");
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithUnsupportedLayout)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NHWC));
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestFusedOp, TestRunWithUnsupportedLayoutLAST)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::LAST));
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestFusedOp, TestRunSuccessWithoutCropPosX)
{
    OpSpec* tmpOpSpec = new OpSpec("testOpSpec");
    tmpOpSpec->AddArg<std::vector<int64_t>>("resize", {resize.first, resize.second});
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", {crop.first, crop.second});
    tmpOpSpec->AddArg<std::string>("interpolation_mode", interpolationMode);
    tmpOpSpec->AddArg<std::string>("round_mode", roundMode);
    tmpOpSpec->AddArg<float>("crop_pos_y", 0.5f);
    tmpOpSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<float>("scale", scale);
    tmpOpSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NCHW));
    tmpOpSpec->AddOutput("testOutput", "cpu");
    ToTensorResizeCropNormalize fusedOp(*tmpOpSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestFusedOp, TestRunSuccessWithoutCropPosY)
{
    OpSpec* tmpOpSpec = new OpSpec("testOpSpec");
    tmpOpSpec->AddArg<std::vector<int64_t>>("resize", {resize.first, resize.second});
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", {crop.first, crop.second});
    tmpOpSpec->AddArg<std::string>("interpolation_mode", interpolationMode);
    tmpOpSpec->AddArg<std::string>("round_mode", roundMode);
    tmpOpSpec->AddArg<float>("crop_pos_x", 0.5f);
    tmpOpSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<float>("scale", scale);
    tmpOpSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NCHW));
    tmpOpSpec->AddOutput("testOutput", "cpu");
    ToTensorResizeCropNormalize fusedOp(*tmpOpSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestFusedOp, TestRunSuccessWithoutInterpolationMode)
{
    OpSpec* tmpOpSpec = new OpSpec("testOpSpec");
    tmpOpSpec->AddArg<std::vector<int64_t>>("resize", {resize.first, resize.second});
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", {crop.first, crop.second});
    tmpOpSpec->AddArg<std::string>("round_mode", roundMode);
    tmpOpSpec->AddArg<float>("crop_pos_x", 0.5f);
    tmpOpSpec->AddArg<float>("crop_pos_y", 0.5f);
    tmpOpSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<float>("scale", scale);
    tmpOpSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NCHW));
    tmpOpSpec->AddOutput("testOutput", "cpu");
    ToTensorResizeCropNormalize fusedOp(*tmpOpSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestFusedOp, TestRunSuccessWithoutRoundMode)
{
    OpSpec* tmpOpSpec = new OpSpec("testOpSpec");
    tmpOpSpec->AddArg<std::vector<int64_t>>("resize", {resize.first, resize.second});
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", {crop.first, crop.second});
    tmpOpSpec->AddArg<std::string>("interpolation_mode", interpolationMode);
    tmpOpSpec->AddArg<float>("crop_pos_x", 0.5f);
    tmpOpSpec->AddArg<float>("crop_pos_y", 0.5f);
    tmpOpSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f, 0.5f});
    tmpOpSpec->AddArg<float>("scale", scale);
    tmpOpSpec->AddArg<int64_t>("layout", static_cast<int64_t>(TensorLayout::NCHW));
    tmpOpSpec->AddOutput("testOutput", "cpu");
    ToTensorResizeCropNormalize fusedOp(*tmpOpSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestFusedOp, TestRunWithInvalidCropSize)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {1LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithInvalidResizeSize)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("resize", {1LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropHeightLargerThanResizeHeight)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {2180LL, 1024LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropWidthLargerThanResizeWidth)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 2180LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropHeightLessThan10)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {5LL, 1024LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropHeightLargerThan8192)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {8193LL, 1024LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropWidthLessThan10)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 5LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropWidthLargerThan8192)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 8193LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithInvalidRoundMode)
{
    PrepareOpSpec();
    opSpec->AddArg<std::string>("round_mode", "invalid_round_mode");
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropPosXLargerThanOne)
{
    PrepareOpSpec();
    opSpec->AddArg<float>("crop_pos_x", 1.5f);
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropPosXLessThanZero)
{
    PrepareOpSpec();
    opSpec->AddArg<float>("crop_pos_x", -0.5f);
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropPosYLargerThanOne)
{
    PrepareOpSpec();
    opSpec->AddArg<float>("crop_pos_y", 1.5f);
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithCropPosYLessThanZero)
{
    PrepareOpSpec();
    opSpec->AddArg<float>("crop_pos_y", -0.5f);
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithoutOpArg)
{
    opSpec = new OpSpec("testOpSpec");
    opSpec->AddOutput("testOutput", "cpu");

    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithInvalidMeanSize)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithInvalidStdSize)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithNegativeStd)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f, -0.5f});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithZeroStd)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("stddev", {0.0f, 0.5f, 1.5f});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestFusedOp, TestRunWithResizeHeightLessThan10)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("resize", {5LL, 1920LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithResizeHeightLargerThan8192)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("resize", {8193LL, 1920LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithResizeWidthLessThan10)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("resize", {1080LL, 5LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestFusedOp, TestRunWithResizeWidthLargerThan8192)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<int64_t>>("resize", {1080LL, 8193LL});
    ToTensorResizeCropNormalize fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

}