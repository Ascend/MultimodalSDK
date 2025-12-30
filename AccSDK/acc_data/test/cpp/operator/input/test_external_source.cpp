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
 * @Date: 2025-4-3 11:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-4-3 11:00:00
 */

#include <random>

#include <gtest/gtest.h>

#include "operator/input/external_source.h"

namespace {
using namespace acclib::accdata;

class TestExternalSource : public ::testing::Test {
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
        if (withOutput) {
            opSpec->AddOutput("testExternalSourceOutput", "testDevice");
        }
    }

    template<typename T>
    void PrepareWorkSpace(bool genData = true, bool withOutput = true)
    {
        size_t tensorSize = 3 * 1080 * 1920;
        std::vector<T> data(tensorSize);
        if (genData) {
            GenerateTensorData<T>(tensorSize, data);
        }

        mInputTensor = std::make_shared<TensorList>(1);
        TensorShape tensorShape = {1, 3, 1080, 1920};
        mInputTensor->operator[](0).Copy<T>(data.data(), tensorShape);
        mInputTensor->operator[](0).SetLayout(TensorLayout::NCHW);
        auto mOutputTensor = std::make_shared<TensorList>(1);
        auto mThreadPool = std::make_shared<ThreadPool>(8, true, "AccData");

        workspace = new Workspace();
        workspace->SetThreadPool(mThreadPool);
        workspace->AddInput(mInputTensor);
        if (withOutput) {
            workspace->AddOutput(mOutputTensor);
        }
    }

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

    OpSpec *opSpec = nullptr;
    Workspace *workspace = nullptr;
    std::shared_ptr<TensorList> mInputTensor;
    std::stringstream buffer;
    std::streambuf *sbuf;
};

TEST_F(TestExternalSource, TestOnceFeedAndOnceRun) // Feed输入一个数据，Run执行一次，运行成功
{
    PrepareOpSpec();
    PrepareWorkSpace<float>();
    ExternalSource externalSource(*opSpec);
    externalSource.Feed(mInputTensor, true);
    EXPECT_EQ(externalSource.Run(*workspace), AccDataErrorCode::H_OK);
}

TEST_F(TestExternalSource, TestThreeTimesFeedAndTwiceRun) // Feed输入三个数据，Run执行两次，运行成功
{
    const uint64_t feedCounts = 3;
    const uint64_t runCounts = 2;
    auto errCode = AccDataErrorCode::H_OK;
    PrepareOpSpec();
    PrepareWorkSpace<float>();
    ExternalSource externalSource(*opSpec);
    for (uint64_t i = 0; i < feedCounts; ++i) {
        externalSource.Feed(mInputTensor, true);
    }
    for (uint64_t i = 0; i < runCounts; ++i) {
        errCode = externalSource.Run(*workspace);
        if (errCode != AccDataErrorCode::H_OK) {
            break;
        }
    }
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestExternalSource, TestTwiceFeedAndThreeTimesRun) // Feed输入两个数据，Run执行三次，运行失败
{
    const uint64_t feedCounts = 2;
    const uint64_t runCounts = 3;
    auto errCode = AccDataErrorCode::H_OK;
    PrepareOpSpec();
    PrepareWorkSpace<float>();
    ExternalSource externalSource(*opSpec);
    for (uint64_t i = 0; i < feedCounts; ++i) {
        externalSource.Feed(mInputTensor, true);
    }
    for (uint64_t i = 0; i < runCounts; ++i) {
        errCode = externalSource.Run(*workspace);
        if (errCode != AccDataErrorCode::H_OK) {
            break;
        }
    }
    EXPECT_EQ(errCode, AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestExternalSource, TestRunOpSpecOutputError) // opSpec NumOutput与workspace NumOutput不一致
{
    PrepareOpSpec(false);
    PrepareWorkSpace<float>(false);
    ExternalSource externalSource(*opSpec);
    EXPECT_EQ(externalSource.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestExternalSource, TestRunExternalSourceEmpty) // ExternalSource为空，运行失败
{
    PrepareOpSpec();
    PrepareWorkSpace<float>(false);
    ExternalSource externalSource(*opSpec);
    EXPECT_EQ(externalSource.Run(*workspace), AccDataErrorCode::H_SINGLEOP_ERROR);
}

TEST_F(TestExternalSource, TestRunGetOutputFailed) // 获取workspace的output失败，运行失败
{
    PrepareOpSpec(false);
    PrepareWorkSpace<float>(true, false);
    ExternalSource externalSource(*opSpec);
    externalSource.Feed(mInputTensor, true);
    EXPECT_EQ(externalSource.Run(*workspace), AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestExternalSource, TestFeedCopySuccess)
{
    PrepareOpSpec();
    PrepareWorkSpace<float>();
    ExternalSource externalSource(*opSpec);
    EXPECT_EQ(externalSource.Feed(mInputTensor, true), AccDataErrorCode::H_OK);
}

TEST_F(TestExternalSource, TestFeedShareDataSuccess)
{
    PrepareOpSpec();
    PrepareWorkSpace<float>();
    ExternalSource externalSource(*opSpec);
    EXPECT_EQ(externalSource.Feed(mInputTensor, false), AccDataErrorCode::H_OK);
}

}
