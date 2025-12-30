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

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "operator/math/normalize.h"

namespace {
using namespace acclib::accdata;

constexpr int PARAM_SCALE = 0;
constexpr int PARAM_ORIGIN_SIZE =  1;
constexpr int PARAM_TENSOR_LAYOUT = 2;

class BaseTestNormalize {
public:
    template<typename T>
    void GenerateTensorData(size_t size, std::vector<T> &data)
    {
        int lowerLimit = 0;
        int upperLimit = 255;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(lowerLimit, upperLimit);

        for (int i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }

    void PrepareOpSpec(bool withOutput = true)
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<std::vector<float>>("mean", {0.458f, 0.458f, 0.458f});
        opSpec->AddArg<std::vector<float>>("stddev", {0.229f, 0.229f, 0.229f});
        opSpec->AddArg<float>("scale", scale);
        if (withOutput) {
            opSpec->AddOutput("testNormalizeOutput", "testDevice");
        }
    }

    template<typename T>
    void PrepareWorkSpace(bool genData = true, bool validLayout = true)
    {
        size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
        std::vector<T> data(tensorSize);
        if (genData) {
            GenerateTensorData<T>(tensorSize, data);
        }

        auto mInputTensor = std::make_shared<TensorList>(1);
        TensorShape tensorShape;
        if (tensorLayout == TensorLayout::NCHW) {
            tensorShape = {1, 3, originImageSize.first, originImageSize.second};
        } else if (tensorLayout == TensorLayout::NHWC) {
            tensorShape = {1, originImageSize.first, originImageSize.second, 3};
        }
        mInputTensor->operator[](0).Copy<T>(data.data(), tensorShape);
        mInputTensor->operator[](0).SetLayout(validLayout ? tensorLayout : TensorLayout::LAST);
        auto mOutputTensor = std::make_shared<TensorList>(1);
        auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

        workspace = new Workspace();
        workspace->SetThreadPool(mThreadPool);
        workspace->AddInput(mInputTensor);
        workspace->AddOutput(mOutputTensor);
    }

    OpSpec *opSpec = nullptr;
    Workspace *workspace = nullptr;
    float scale {1.0f};
    std::pair<int64_t, int64_t> originImageSize = std::make_pair(1080, 1920);
    TensorLayout tensorLayout = TensorLayout::NCHW;
};

class ParamTestNormalize : public ::testing::TestWithParam<std::tuple<float, std::pair<int64_t, int64_t>,
        TensorLayout>>, public BaseTestNormalize {
public:
    void SetUp()
    {
        scale = std::get<PARAM_SCALE>(GetParam());
        originImageSize = std::get<PARAM_ORIGIN_SIZE>(GetParam());
        tensorLayout = std::get<PARAM_TENSOR_LAYOUT>(GetParam());
        PrepareOpSpec();
        PrepareWorkSpace<float>();
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
    TestNormalizeCases,
    ParamTestNormalize,
    ::testing::Values(
        // scale originSize tensorLayout
        std::make_tuple(1.0f, std::make_pair(1080, 1920), TensorLayout::NCHW),
        std::make_tuple(0.5f, std::make_pair(1080, 1920), TensorLayout::NCHW),
        std::make_tuple(1.0f, std::make_pair(1080, 1920), TensorLayout::NHWC),
        std::make_tuple(0.5f, std::make_pair(1080, 1920), TensorLayout::NHWC)
        ),
    [](const testing::TestParamInfo<ParamTestNormalize::ParamType>& info) {
        float scale = std::get<PARAM_SCALE>(info.param);
        auto strScale = (scale == 1.0 ? "notScaled" : "scaled");
        auto tensorLayout = std::get<PARAM_TENSOR_LAYOUT>(info.param);
        std::ostringstream oss;
        oss << "layout_" << tensorLayout << "_" << strScale;
        return oss.str();
    }
);

TEST_P(ParamTestNormalize, TestRunSuccess)
{
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), tensorLayout);
}

class TestNormalize : public ::testing::Test, public BaseTestNormalize {
public:
    void SetUp()
    {
        buffer.str(std::string());  // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }

    void TearDown()
    {
        workspace->Clear();
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

TEST_F(TestNormalize, TestRunOpSpecOutputError) // opSpec NumOutput与workspace NumOutput不一致
{
    PrepareOpSpec(false);
    PrepareWorkSpace<float>(false);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestNormalize, TestRunInputTypeError) // 输入数据类型非FP32
{
    PrepareOpSpec();
    PrepareWorkSpace<uint8_t>();
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestNormalize, TestRunLayoutError) // layout非法
{
    PrepareOpSpec();
    PrepareWorkSpace<float>(true, false);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunMeanDimError) // mean维度与channel数不一致.channel=3
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("mean", {0.458f, 0.458f});
    PrepareWorkSpace<float>();
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunStddevDimError) // stddev维度与channel数不一致.channel=3
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("stddev", {0.229f, 0.229f});
    PrepareWorkSpace<float>();
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithoutMean)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("stddev", {0.501f, 0.501f, 0.501f});
    operatorSpec->AddArg<float>("scale", scale);
    operatorSpec->AddOutput("testNormalizeOutput", "testDevice");
    PrepareWorkSpace<float>();

    Normalize normalize(*operatorSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestNormalize, TestRunWithoutStddev)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("mean", {0.354f, 0.354f, 0.354f});
    operatorSpec->AddArg<float>("scale", scale);
    operatorSpec->AddOutput("testNormalizeOutput", "testDevice");
    PrepareWorkSpace<float>();

    Normalize normalize(*operatorSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestNormalize, TestRunWithoutScale)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("mean", {0.354f, 0.354f, 0.354f});
    operatorSpec->AddArg<std::vector<float>>("stddev", {0.501f, 0.501f, 0.501f});
    operatorSpec->AddOutput("testNormalizeOutput", "testDevice");
    PrepareWorkSpace<float>();

    Normalize normalize(*operatorSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestNormalize, TestRunWithStddevZero)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("mean", {0.354f, 0.354f, 0.354f});
    operatorSpec->AddArg<std::vector<float>>("stddev", {0.0f, 0.0f, 0.0f});
    operatorSpec->AddOutput("testNormalizeOutput", "testDevice");
    PrepareWorkSpace<float>();

    Normalize normalize(*operatorSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithNegativeStddev)
{
    auto operatorSpec = std::make_shared<OpSpec>("testOpSpec");
    operatorSpec->AddArg<std::vector<float>>("mean", {0.354f, 0.354f, 0.354f});
    operatorSpec->AddArg<std::vector<float>>("stddev", {-3.2f, -3.2f, -3.2f});
    operatorSpec->AddOutput("testNormalizeOutput", "testDevice");
    PrepareWorkSpace<float>();

    Normalize normalize(*operatorSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithNoTensorList)
{
    PrepareOpSpec();

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestNormalize, TestRunWithEmptyTensorList)
{
    PrepareOpSpec();

    auto mInputTensor = std::make_shared<TensorList>();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithDifferentLayoutTensor)
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * height * width;
    std::vector<float> data(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorListLength);
    TensorShape tensorShape = {1, 3, height, width};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NHWC);

    auto mOutputTensor = std::make_shared<TensorList>(tensorListLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithDifferentDataTypeTensor)
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * height * width;
    std::vector<float> data(tensorSize);
    std::vector<int8_t> data1(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorListLength);
    TensorShape tensorShape = {1, 3, height, width};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<int8_t>(data1.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorListLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithDifferentShapeTensor)
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * height * width;
    std::vector<float> data(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorListLength);
    TensorShape tensorShape = {1, 3, height, width};
    TensorShape tensorShape1 = {1, height, width, 3};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<float>(data.data(), tensorShape1);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorListLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

// outputList Resize input.Shape int normalize so succeed
TEST_F(TestNormalize, TestRunWithInputExceedOutputTensors)
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * height * width;
    std::vector<float> data(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorListLength);
    TensorShape tensorShape = {1, 3, height, width};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    auto errCode = AccDataErrorCode::H_OK;
    auto &inputList = workspace->GetInput(0, errCode);
    auto &outputList = workspace->GetOutput(0, errCode);

    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestNormalize, TestRunWithOutputExceedInputTensors)
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * height * width;
    std::vector<float> data(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 3, height, width};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorListLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestNormalize, TestRunWithThreeDimsShape)
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * height * width;
    std::vector<float> data(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {3, height, width};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestNormalize, TestRunWithInvalidChannel) // Single channel is invalid
{
    int height = 1080;
    int width = 1920;
    int tensorListLength = 2;
    PrepareOpSpec();

    size_t tensorSize = 1 * height * width;
    std::vector<float> data(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 1, height, width};
    mInputTensor->operator[](0).Copy<float>(data.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    Normalize normalize(*opSpec);
    EXPECT_EQ(normalize.Run(*workspace), AccDataErrorCode::H_TENSOR_ERROR);
}
}