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

#include "operator/fusion/qwen_fusion_ops.h"

namespace {
using namespace acclib::accdata;

class BaseTestQwenFusedOp {
protected:
    template<typename T>
    void GenerateTensorDatas(size_t size, std::vector<T> &datas)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

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
        opSpec->AddArg<std::vector<float>>("mean", mean);
        opSpec->AddArg<std::vector<float>>("stddev", stddev);
        opSpec->AddArg<int64_t>("min_pixels", minPixels);
        opSpec->AddArg<int64_t>("max_pixels", maxPixels);
        opSpec->AddArg<int64_t>("patch_size", patchSize);
        opSpec->AddArg<int64_t>("temporal_patch_size", temporalPatchSzie);
        opSpec->AddArg<int64_t>("merge_size", mergeSize);
        if (withOutput) {
            opSpec->AddOutput("testOutput", "cpu");
        }
    }

    OpSpec* opSpec = nullptr;
    Workspace* workspace = nullptr;
    std::pair<int64_t, int64_t> originImageSize = std::make_pair(1920LL, 1080LL);
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> stddev = {0.5f, 0.5f, 0.5f};
    int64_t minPixels = 56 * 56;
    int64_t maxPixels = 28 * 28 * 1280;
    int64_t patchSize = 14;
    int64_t temporalPatchSzie = 2;
    int64_t mergeSize = 2;
};

class TestQwenFusedOp : public ::testing::Test, public BaseTestQwenFusedOp {
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

TEST_F(TestQwenFusedOp, TestRunWithWrongTensorList)
{
    PrepareOpSpec();
    workspace->Clear();
    auto mInputTensor = std::make_shared<TensorList>();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestQwenFusedOp, TestRunWithoutInputs)
{
    PrepareOpSpec();
    workspace->Clear();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithoutOutputs)
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

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithUnsupportedDataType)
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

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestQwenFusedOp, TestRunWithUnsupportedDataLayout)
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

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_FUSIONOP_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidDataLayout)
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

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidInputDataChannel) // Single channel is invalid
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 1, originImageSize.first, originImageSize.second};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidMean)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("mean", {0.5f, 0.5f});
    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidStd)
{
    PrepareOpSpec();
    opSpec->AddArg<std::vector<float>>("stddev", {0.5f, 0.5f});
    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidMinPixels)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("min_pixels", 9 * 9);
    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    opSpec->AddArg<int64_t>("min_pixels", 4097 * 4097);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidMaxPixels)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("max_pixels", 9 * 9);
    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    opSpec->AddArg<int64_t>("max_pixels", 4097 * 4097);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithInvalidMinMaxPixels)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("min_pixels", 1920);
    opSpec->AddArg<int64_t>("max_pixels", 1080);
    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithTemporalPatchSize)
{
    PrepareOpSpec();
    opSpec->AddArg<int64_t>("temporal_patch_size", 4);
    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestQwenFusedOp, TestRunWithSmallInput)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * 9 * 9;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 9, 9, 3};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestQwenFusedOp, TestRunWithBigInput)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * 8193 * 8193;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, 8193, 8193, 3};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestQwenFusedOp, TestRunSuccess)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, originImageSize.first, originImageSize.second, 3};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_EQ(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
    EXPECT_EQ((*mOutputTensor)[0].Layout(), TensorLayout::PLAIN);
}

TEST_F(TestQwenFusedOp, TestInvalidChannel)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 1 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {1, originImageSize.first, originImageSize.second, 1};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestQwenFusedOp, TestInvalidNumSamples)
{
    PrepareOpSpec();
    workspace->Clear();
    size_t tensorSize = 3 * 3 * originImageSize.first * originImageSize.second;
    std::vector<uint8_t> datas(tensorSize);
    GenerateTensorDatas<uint8_t>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = {3, originImageSize.first, originImageSize.second, 3};
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    QwenFusionOp fusedOp(*opSpec);
    EXPECT_NE(fusedOp.Run(*workspace), AccDataErrorCode::H_OK);
}

}