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
#include "random"

#include "operator/image/to_tensor.h"

namespace {
using namespace acclib::accdata;

constexpr int PIXEL = 255;

class TestToTensor : public ::testing::Test {
public:
    void PrepareOpSpec(TensorLayout layout = TensorLayout::NHWC, bool withOutput = true)
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<int64_t>("layout", static_cast<int64_t>(layout));
        if (withOutput) {
            opSpec->AddOutput("testToTensorOutput", "cpu");
        }
    }

    template<typename T>
    void PrepareWorkSpace(TensorLayout tensorLayout = TensorLayout::NHWC, int inputTensorSize = 1)
    {
        size_t tensorSize = 3 * 1080 * 1920;
        TensorShape tensorShape = {1, 1080, 1920, 3};
        if (tensorLayout == TensorLayout::NCHW) {
            tensorShape = {1, 3, 1080, 1920};
        }

        std::vector<T> datas(tensorSize);
        GenerateTensorDatas<T>(tensorSize, datas);

        auto mInputTensor = std::make_shared<TensorList>(inputTensorSize);

        if (inputTensorSize) {
            mInputTensor->operator[](0).Copy<T>(datas.data(), tensorShape);
            mInputTensor->operator[](0).SetLayout(tensorLayout);
        }

        auto mOutputTensor = std::make_shared<TensorList>(1);
        auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

        workspace = new Workspace();
        workspace->SetThreadPool(mThreadPool);
        workspace->AddInput(mInputTensor);
        workspace->AddOutput(mOutputTensor);
    }

    void SetUp()
    {
        Logger::SetLogLevelStr("error");  // capture debug info
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

    OpSpec *opSpec = nullptr;
    Workspace *workspace = nullptr;
    std::stringstream buffer;
    std::streambuf *sbuf;
};

TEST_F(TestToTensor, RunSuccessNHWCToNHWC)
{
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NHWC);
}

TEST_F(TestToTensor, RunSuccessNCHWToNHWC)
{
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NHWC);
}

TEST_F(TestToTensor, RunSuccessNHWCToNCHW)
{
    PrepareOpSpec(TensorLayout::NCHW);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NCHW);
}

TEST_F(TestToTensor, RunSuccessNCHWToNCHW)
{
    PrepareOpSpec(TensorLayout::NCHW);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NCHW);
}

TEST_F(TestToTensor, RunWithInconsistentOutputs)
{
    PrepareOpSpec(TensorLayout::NCHW, false);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestToTensor, RunWithUnsupportedInputTensorLayout)
{
    PrepareOpSpec(TensorLayout::NCHW);
    PrepareWorkSpace<uint8_t>(TensorLayout::LAST);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensor, RunWithUnsupportedOutputTensorLayout)
{
    PrepareOpSpec(TensorLayout::LAST);
    PrepareWorkSpace<uint8_t>();

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensor, RunWithDefaultTensor) // default tensor shape.size is 0
{
    PrepareOpSpec(TensorLayout::NCHW);
    workspace = new Workspace();
    auto mInputTensor = std::make_shared<TensorList>(1);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensor, RunWithEmptyTensorList)
{
    PrepareOpSpec(TensorLayout::NCHW);
    workspace = new Workspace();
    auto mInputTensor = std::make_shared<TensorList>(0);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensor, RunWithoutWorkSpaceInput)
{
    PrepareOpSpec(TensorLayout::NCHW);
    workspace = new Workspace();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestToTensor, RunWithUnsupportedLayout)
{
    PrepareOpSpec(TensorLayout::LAST);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensor, RunWithUnsupportedLayoutAndDataLayout)
{
    PrepareOpSpec(TensorLayout::LAST);
    PrepareWorkSpace<uint8_t>(TensorLayout::LAST);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensor, RunWithoutOutput)
{
    PrepareOpSpec(TensorLayout::NCHW, false);

    workspace = new Workspace();

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 1080, 1920, 3};

    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestToTensor, RunWithFloatTensor)
{
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<float>(TensorLayout::NHWC);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensor, RunWithEmptyOpSpec)
{
    opSpec = new OpSpec("EmptyOpSpec");
    ToTensor toTensor(*opSpec);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestToTensor, RunWithEmptyWorkSpace)
{
    PrepareOpSpec(TensorLayout::NHWC);
    ToTensor toTensor(*opSpec);
    workspace = new Workspace();

    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestToTensor, RunWithRedundantOpSpecArgs)
{
    PrepareOpSpec(TensorLayout::NHWC);
    opSpec->AddArg<bool>("RedundantOpSpecArgs", true);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensor, RunWithRedundantWorkSpaceInput)
{
    int inputTensorSize = 32;
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    auto redundantInputTensor = std::make_shared<TensorList>(inputTensorSize);
    workspace->AddInput(redundantInputTensor);
    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensor, RunWithRedundantOpSpecOutput)
{
    PrepareOpSpec(TensorLayout::NHWC);
    opSpec->AddOutput("RedundantOpSpecArgs", "true");
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestToTensor, RunWithRedundantWorkSpaceOutput)
{
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestToTensor, RunWithThreeDimTensorShape)
{
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 3 * 1920;
    TensorShape tensorShape = {1, 3, 1920};

    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensor, RunWithDifferentShapeTensor)
{
    int tensorNumber = 2;
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 1080, 1920, 3};
    TensorShape tensorShape1 = {3, 1, 1080, 1920};

    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(tensorNumber);
    (*mInputTensor)[0].Copy<uint8_t>(datas.data(), tensorShape);
    (*mInputTensor)[0].SetLayout(TensorLayout::NCHW);

    (*mInputTensor)[1].Copy<uint8_t>(datas.data(), tensorShape1);
    (*mInputTensor)[1].SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorNumber);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensor, RunWithDifferentDataTypeTensor)
{
    int tensorNumber = 2;
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 1080, 1920, 3};

    std::vector<uint8_t> datas(tensorSize);
    std::vector<int8_t> datas1(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorNumber);
    (*mInputTensor)[0].Copy<uint8_t>(datas.data(), tensorShape);
    (*mInputTensor)[0].SetLayout(TensorLayout::NCHW);

    (*mInputTensor)[1].Copy<int8_t>(datas1.data(), tensorShape);
    (*mInputTensor)[1].SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorNumber);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensor, RunWithDifferentLayoutTensor)
{
    int tensorNumber = 2;
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 3, 1080, 1920};

    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(tensorNumber);
    (*mInputTensor)[0].Copy<uint8_t>(datas.data(), tensorShape);
    (*mInputTensor)[0].SetLayout(TensorLayout::NHWC);

    (*mInputTensor)[1].Copy<uint8_t>(datas.data(), tensorShape);
    (*mInputTensor)[1].SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorNumber);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

// input tensorlist and output tensorlist have different number
TEST_F(TestToTensor, RunWithInputExceedsOutputTensors)
{
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 3, 1080, 1920};

    std::vector<uint8_t> datas(tensorSize, 100U);

    auto mInputTensor = std::make_shared<TensorList>(2);
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestToTensor, RunWithOutputExceedInputTensors)
{
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 3, 1080, 1920};

    std::vector<uint8_t> datas(tensorSize, 100U);

    auto mInputTensor = std::make_shared<TensorList>(1);
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(2);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_EQ(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensor, RunWithInvalidChannelTensor) // Single channel is invalid
{
    PrepareOpSpec(TensorLayout::NCHW);

    size_t tensorSize = 1 * 1080 * 1920;
    TensorShape tensorShape = {1, 1, 1080, 1920};

    std::vector<uint8_t> datas(tensorSize, 100U);

    auto mInputTensor = std::make_shared<TensorList>(1);
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(2);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ToTensor toTensor(*opSpec);
    EXPECT_NE(toTensor.Run(*workspace), AccDataErrorCode::H_OK);
}

}