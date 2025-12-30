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

#include "operator/image/to_tensor_args.h"

namespace {
using namespace acclib::accdata;

constexpr int PIXEL = 255;

class TestToTensorArgs : public ::testing::Test {
public:
    void PrepareOpSpec(TensorLayout layout = TensorLayout::NHWC)
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<int64_t>("layout", static_cast<int64_t>(layout));
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

TEST_F(TestToTensorArgs, SetUpSuccessNHWCToNHWC)
{
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensorArgs, SetUpSuccessNCHWToNHWC)
{
    PrepareOpSpec(TensorLayout::NHWC);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensorArgs, SetUpSuccessNHWCToNCHW)
{
    PrepareOpSpec(TensorLayout::NCHW);
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensorArgs, SetUpSuccessNCHWToNCHW)
{
    PrepareOpSpec(TensorLayout::NCHW);
    PrepareWorkSpace<uint8_t>(TensorLayout::NCHW);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestToTensorArgs, SetUpErrorOutputLayout)
{
    PrepareOpSpec(TensorLayout::LAST);
    PrepareWorkSpace<uint8_t>();

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensorArgs, SetUpInvalidOutputLayout)
{
    PrepareOpSpec();
    PrepareWorkSpace<uint8_t>();

    ToTensorArgs toTensorArgs;
    int64_t invalidLayout = 100;
    opSpec->AddArg<int64_t>("layout", invalidLayout);
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensorArgs, SetUpInvalidInputLayout)
{
    PrepareOpSpec();
    PrepareWorkSpace<uint8_t>(TensorLayout::LAST);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensorArgs, SetUpEmptyTensorList)
{
    PrepareOpSpec();
    PrepareWorkSpace<uint8_t>(TensorLayout::NHWC, 0);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensorArgs, SetUpErrorTensorDataType)
{
    PrepareOpSpec();
    PrepareWorkSpace<float>();

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensorArgs, SetUpWithoutLayout)
{
    opSpec = new OpSpec("testOpSpec");
    PrepareWorkSpace<uint8_t>();

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestToTensorArgs, SetUpWithDifferentDataTypeTensor)
{
    int inputTensorSize = 2;
    PrepareOpSpec();

    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 1080, 1920, 3};

    std::vector<uint8_t> datas(tensorSize);
    std::vector<int8_t> datas1(tensorSize);

    auto inputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto outputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto threadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    inputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    inputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    inputTensor->operator[](1).Copy<int8_t>(datas1.data(), tensorShape);
    inputTensor->operator[](1).SetLayout(TensorLayout::NHWC);

    workspace = new Workspace();
    workspace->SetThreadPool(threadPool);
    workspace->AddInput(inputTensor);
    workspace->AddOutput(outputTensor);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensorArgs, SetUpWithDifferentShapeTensor)
{
    int inputTensorSize = 2;
    PrepareOpSpec();
    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 1080, 1920, 3};
    TensorShape tensorShape1 = {1, 3, 1080, 1920};

    std::vector<uint8_t> datas(tensorSize, 0);

    auto inputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto outputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto threadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    inputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    inputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    inputTensor->operator[](1).Copy<uint8_t>(datas.data(), tensorShape1);
    inputTensor->operator[](1).SetLayout(TensorLayout::NHWC);

    workspace = new Workspace();
    workspace->SetThreadPool(threadPool);
    workspace->AddInput(inputTensor);
    workspace->AddOutput(outputTensor);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensorArgs, SetUpWithDifferentLayoutTensor)
{
    int inputTensorSize = 2;
    PrepareOpSpec();
    size_t tensorSize = 3 * 1080 * 1920;
    TensorShape tensorShape = {1, 1080, 1920, 3};

    std::vector<uint8_t> datas(tensorSize, 0);

    auto inputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto outputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto threadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    inputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    inputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    inputTensor->operator[](1).Copy<uint8_t>(datas.data(), tensorShape);
    inputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    workspace = new Workspace();
    workspace->SetThreadPool(threadPool);
    workspace->AddInput(inputTensor);
    workspace->AddOutput(outputTensor);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestToTensorArgs, SetUpWithoutInputTensor)
{
    PrepareOpSpec();

    workspace = new Workspace();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddOutput(mOutputTensor);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestToTensorArgs, SetUpWithInvalidShapeTensor)
{
    int inputTensorSize = 1;
    PrepareOpSpec();
    size_t tensorSize = 3 * 1080;
    TensorShape tensorShape = {1, 1080, 3};

    std::vector<uint8_t> datas(tensorSize, 0);

    auto inputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto outputTensor = std::make_shared<TensorList>(inputTensorSize);
    auto threadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    inputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    inputTensor->operator[](0).SetLayout(TensorLayout::NHWC);

    workspace = new Workspace();
    workspace->SetThreadPool(threadPool);
    workspace->AddInput(inputTensor);
    workspace->AddOutput(outputTensor);

    ToTensorArgs toTensorArgs;
    EXPECT_EQ(toTensorArgs.Setup(*opSpec, *workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}
}