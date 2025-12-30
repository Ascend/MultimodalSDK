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
 * Create: 2025-03-26
 */
#include <gtest/gtest.h>

#include "random"

#include "operator/image/resize_crop.h"
#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"
#include "common/thread_pool.h"
#include "interface/accdata_tensor.h"
#include "interface/logger.h"

namespace {
using namespace acclib::accdata;

class TestResizeCrop : public ::testing::Test {
public:
    template <typename T> void GenerateTensorDatas(size_t size, std::vector<T> &datas)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (int i = 0; i < size; ++i) {
            datas[i] = dis(gen);
        }
    }

    void PrepareOpSpec()
    {
        opSpec = new OpSpec("testOpSpec");
        opSpec->AddArg<std::vector<int64_t>>("resize", { 1080LL, 1920LL });
        opSpec->AddArg<std::vector<int64_t>>("crop", { 1024LL, 1024LL });
        opSpec->AddArg<std::string>("interpolation_mode", "bilinear");
        opSpec->AddArg<std::string>("round_mode", "truncate");
        opSpec->AddOutput("testOutput", "testDevice");
    }

    template <typename T> void PrepareWorkSpace()
    {
        size_t tensorSize = 3 * 1080 * 1920;
        std::vector<T> datas(tensorSize);
        GenerateTensorDatas<T>(tensorSize, datas);

        auto mInputTensor = std::make_shared<TensorList>(1);
        TensorShape tensorShape = { 1, 3, 1080, 1920 };
        mInputTensor->operator[](0).Copy<T>(datas.data(), tensorShape);
        mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
        auto mOutputTensor = std::make_shared<TensorList>(1);
        auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

        workspace = new Workspace();
        workspace->SetThreadPool(mThreadPool);
        workspace->AddInput(mInputTensor);
        workspace->AddOutput(mOutputTensor);
    }

    void SetUp()
    {
        PrepareOpSpec();
        Logger::SetLogLevelStr("debug"); // capture debug info
        buffer.str(std::string());       // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }

    void TearDown()
    {
        workspace->Clear();
        OpSpec *opSpecDeleter = opSpec;
        delete opSpecDeleter;
        Workspace *workspaceDeleter = workspace;
        delete workspaceDeleter;
        opSpec = nullptr;
        workspace = nullptr;

        Logger::SetLogLevelStr("info");
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;
    }

    OpSpec *opSpec = nullptr;
    Workspace *workspace = nullptr;
    std::stringstream buffer;
    std::streambuf *sbuf;
};

TEST_F(TestResizeCrop, TestRunSuccessBilinearInterpolation) // 运行成功无报错，插值类型Bilinear
{
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NCHW);
}

TEST_F(TestResizeCrop, TestRunSuccessBicubicInterpolation) // 运行成功无报错，插值类型Bicubic
{
    PrepareWorkSpace<float>();
    opSpec->AddArg<std::string>("interpolation_mode", "bicubic");
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
    auto errCode = AccDataErrorCode::H_OK;
    auto &output = workspace->GetOutput(0, errCode);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    EXPECT_EQ(output[0].Layout(), TensorLayout::NCHW);
}

TEST_F(TestResizeCrop, TestRunSuccessDefaultInterpolation) // 运行成功无报错，插值类型未手动指定，使用default值
{
    OpSpec *tmpOpSpec = new OpSpec("testOpSpec");
    tmpOpSpec->AddArg<std::vector<int64_t>>("resize", { 1080LL, 1920LL });
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", { 1024LL, 1024LL });
    tmpOpSpec->AddArg<std::string>("round_mode", "truncate");
    tmpOpSpec->AddArg<float>("crop_pos_x", 0.5f);
    tmpOpSpec->AddArg<float>("crop_pos_y", 0.5f);
    tmpOpSpec->AddOutput("testOutput", "testDevice");
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*tmpOpSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestResizeCrop, TestRunInvalidInterpolationMode) // 插值类型非法
{
    PrepareWorkSpace<float>();
    opSpec->AddArg<std::string>("interpolation_mode", "Invalid");
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestRunSuccessRound) // 运行成功无报错，round类型Round
{
    PrepareWorkSpace<float>();
    opSpec->AddArg<std::string>("round_mode", "round");
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestResizeCrop, TestRunSuccessTruncate) // 运行成功无报错，round类型Truncate
{
    PrepareWorkSpace<float>();
    opSpec->AddArg<std::string>("round_mode", "truncate");
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestResizeCrop, TestRunSuccessDefaultRound) // 运行成功无报错，round类型未手动指定，使用default值
{
    OpSpec *tmpOpSpec = new OpSpec("testOpSpec");
    tmpOpSpec->AddArg<std::vector<int64_t>>("resize", { 1080LL, 1920LL });
    tmpOpSpec->AddArg<std::vector<int64_t>>("crop", { 1024LL, 1024LL });
    tmpOpSpec->AddArg<std::string>("interpolation_mode", "bilinear");
    tmpOpSpec->AddArg<float>("crop_pos_x", 0.5f);
    tmpOpSpec->AddArg<float>("crop_pos_y", 0.5f);
    tmpOpSpec->AddOutput("testOutput", "testDevice");
    PrepareWorkSpace<float>();
    tmpOpSpec->AddArg<std::string>("interpolation_mode", "bicubic");
    ResizeCrop resizeCrop(*tmpOpSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
    delete tmpOpSpec;
    tmpOpSpec = nullptr;
}

TEST_F(TestResizeCrop, TestRunInvalidRoundMode) // round类型非法
{
    PrepareWorkSpace<float>();
    opSpec->AddArg<std::string>("round_mode", "Invalid");
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestRunEmptyInput) // 未输入tensorlist
{
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace = new Workspace();
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestResizeCrop, TestRunEmptyInputTensor) // 输入tensorlist为空
{
    auto mInputTensor = std::make_shared<TensorList>();
    auto mOutputTensor = std::make_shared<TensorList>(1);
    workspace = new Workspace();
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestResizeCrop, TestRunLayoutError) // 输入tensor layout错误，当前仅支持nchw
{
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);
    GenerateTensorDatas<float>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 1, 1920, 1080, 3 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);

    workspace = new Workspace();
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestResizeCrop, TestRunInputTypeError) // 输入tensor数据类型错误，当前仅支持fp32
{
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<uint8_t> datas(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 1, 1920, 1080, 3 };
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);

    workspace = new Workspace();
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestResizeCrop, TestRunInConsistentOutputSize) //  output size 错误, workspace与opSpec output数量不一致
{
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<uint8_t> datas(tensorSize);
    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 1, 1920, 1080, 3 };
    mInputTensor->operator[](0).Copy<uint8_t>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NHWC);
    auto mOutputTensor = std::make_shared<TensorList>(1);

    workspace = new Workspace();
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);

    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestResizeCrop, TestCropHeightGreaterThanResizeHeight) // crop的height超过resize的height
{
    opSpec->AddArg<std::vector<int64_t>>("crop", { 2160LL, 1024LL });
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestCropWidthGreaterThanResizeWidth) // crop的width超过resize的width
{
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 2160LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestCropHeightLessThan10)
{
    opSpec->AddArg<std::vector<int64_t>>("crop", {5LL, 1024LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestCropHeightLargerThan8192)
{
    opSpec->AddArg<std::vector<int64_t>>("crop", {8193LL, 1024LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestCropWidthLessThan10)
{
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 5LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestCropWidthLargerThan8192)
{
    opSpec->AddArg<std::vector<int64_t>>("crop", {1024LL, 8193LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestResizeHeightLessThan10) // resize的height小于10
{
    opSpec->AddArg<std::vector<int64_t>>("resize", {5LL, 1920LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestResizeHeightLargerThan8192) // resize的height大于8192
{
    opSpec->AddArg<std::vector<int64_t>>("resize", {8193LL, 1920LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestResizeWidthLessThan10) // resize的width小于10
{
    opSpec->AddArg<std::vector<int64_t>>("resize", {1280LL, 5LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestInputLenghtLargerThan8192) // resize的输入大于8192
{
    size_t tensorSize = 3 * 1080 * 8193;
    std::vector<float> datas(tensorSize);
    GenerateTensorDatas<float>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 1, 3, 1080, 8193 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestResizeCrop, TestInputLenghtLessThan10) // resize的输入小于10
{
    size_t tensorSize = 3 * 1080 * 9;
    std::vector<float> datas(tensorSize);
    GenerateTensorDatas<float>(tensorSize, datas);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 1, 3, 1080, 9 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_TENSOR_ERROR);
}


TEST_F(TestResizeCrop, TestResizeWidthLargerThan8192) // resize的width大于8192
{
    opSpec->AddArg<std::vector<int64_t>>("resize", {1280LL, 8193LL});
    PrepareWorkSpace<float>();
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestResizeCrop, TestResizeCropWithDifferentShapeTensor)
{
    int tensorLength = 2;
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorLength);
    TensorShape tensorShape = { 1, 3, 1080, 1920 };
    TensorShape tensorShape1 = { 3, 1, 1080, 1920 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<float>(datas.data(), tensorShape1);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestResizeCrop, TestResizeCropWithDifferentDataTypeTensor)
{
    int tensorLength = 2;
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);
    std::vector<int> datas1(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorLength);
    TensorShape tensorShape = { 1, 3, 1080, 1920 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<int>(datas1.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestResizeCrop, TestResizeCropWithDifferentLayoutTensor)
{
    int tensorLength = 2;
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorLength);
    TensorShape tensorShape = { 1, 3, 1080, 1920 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NHWC);

    auto mOutputTensor = std::make_shared<TensorList>(tensorLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}

TEST_F(TestResizeCrop, TestResizeCropWithOutputExceedInputTensors)
{
    int tensorLength = 2;
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 1, 3, 1080, 1920 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(tensorLength);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestResizeCrop, TestResizeCropWithInputExceedOutputTensors)
{
    int tensorLength = 2;
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(tensorLength);
    TensorShape tensorShape = { 1, 3, 1080, 1920 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
    mInputTensor->operator[](1).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](1).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}
TEST_F(TestResizeCrop, TestResizeCropWithThreeDimShapeTensors)
{
    int tensorLength = 2;
    size_t tensorSize = 3 * 1080 * 1920;
    std::vector<float> datas(tensorSize);

    auto mInputTensor = std::make_shared<TensorList>(1);
    TensorShape tensorShape = { 3, 1080, 1920 };
    mInputTensor->operator[](0).Copy<float>(datas.data(), tensorShape);
    mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);

    auto mOutputTensor = std::make_shared<TensorList>(1);
    auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

    workspace = new Workspace();
    workspace->SetThreadPool(mThreadPool);
    workspace->AddInput(mInputTensor);
    workspace->AddOutput(mOutputTensor);
    ResizeCrop resizeCrop(*opSpec);
    EXPECT_EQ(resizeCrop.Run(*workspace), AccDataErrorCode::H_COMMON_INVALID_PARAM);
}
}
