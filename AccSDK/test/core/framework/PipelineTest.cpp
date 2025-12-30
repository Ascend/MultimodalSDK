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
* Description: test pipeline api.
* Author: ACC SDK
* Create: 2025
* History: NA
*/

#include <cstdint>
#include <gtest/gtest.h>
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/core/framework/Pipeline.h"

using namespace Acc;
using namespace acclib::accdata;
namespace {
constexpr int VALID_THREAD_NUM = 2;
constexpr int INVALID_THREAD_NUM = 100000;
constexpr int DEFAULT_PIPELINE_BATCH_SIZE = 1;
constexpr int DEFAULT_PIPELINE_DEPTH = 2;
constexpr size_t BATCH_SIZE = 1;
constexpr size_t HEIGHT = 10;
constexpr size_t WIDTH = 10;
constexpr size_t CHANNEL = 3;
constexpr float UNIFORM_VALUE_FLOAT = 128.0f;
constexpr int8_t UNIFORM_VALUE_INT8 = 127;
constexpr float DEFAULT_MEAN = 0.5f;
constexpr float DEFAULT_STD = 0.5f;

class PipelineTest : public testing::Test {
protected:
    void SetUp() override
    {
        RegisterLogConf(LogLevel::WARN, nullptr);
    }
};

TEST_F(PipelineTest, Test_Pipeline_Default_Constructor_Should_Success)
{
    EXPECT_NO_THROW({
        Pipeline pipeline;
    });
}

TEST_F(PipelineTest, Test_Pipeline_Constructor_With_Valid_NumThreads_And_EnableFusion_Should_Success)
{
    EXPECT_NO_THROW({
        Pipeline pipeline(VALID_THREAD_NUM, true);
    });
}

TEST_F(PipelineTest, Test_Pipeline_Constructor_With_Invalid_NumThreads_Should_Fail)
{
    EXPECT_THROW({
        Pipeline pipeline(INVALID_THREAD_NUM, false);
    }, std::runtime_error);
}

TEST_F(PipelineTest, Test_Pipeline_Build_With_Unsupported_OpSpecs_Should_Fail)
{
    Pipeline pipeline(VALID_THREAD_NUM, true);
    auto invalidOp = AccDataOpSpec::Create("InvalidOpName");
    invalidOp->AddOutput("InvalidOpOutput", "cpu");
    int ret = pipeline.Build({invalidOp}, "InvalidOpOutput");
    EXPECT_EQ(ret, ERR_ACC_DATA_EXECUTE_FAILURE);
}

TEST_F(PipelineTest, Test_Pipeline_Build_With_Invalid_Output_Should_Fail)
{
    Pipeline pipeline(VALID_THREAD_NUM, true);
    auto externalInput = AccDataOpSpec::Create("ExternalSource");
    externalInput->AddOutput("ExternalSourceOutput", "cpu");
    int ret = pipeline.Build({externalInput}, "InvalidOutput");
    EXPECT_EQ(ret, ERR_ACC_DATA_EXECUTE_FAILURE);
}

TEST_F(PipelineTest, Test_Pipeline_Run_Normalize_Should_Success)
{
    // construct input tensor
    std::vector<size_t> tensorShape = {BATCH_SIZE, HEIGHT, WIDTH, CHANNEL};
    const size_t numElements = BATCH_SIZE * HEIGHT * WIDTH * CHANNEL;
    std::vector<float> inputData(numElements, UNIFORM_VALUE_FLOAT);
    auto inputDataPtr = static_cast<void*>(inputData.data());
    Tensor inputTensor(inputDataPtr, tensorShape, DataType::FLOAT32, TensorFormat::NHWC, "cpu");

    // construct pipeline for normalize
    Pipeline pipeline(VALID_THREAD_NUM, true);
    auto externalInput = AccDataOpSpec::Create("ExternalSource");
    externalInput->AddOutput("ExternalSourceOutput", "cpu");
    auto normalize = AccDataOpSpec::Create("Normalize");
    std::vector<float> mean = {DEFAULT_MEAN, DEFAULT_MEAN, DEFAULT_MEAN};
    std::vector<float> std = {DEFAULT_STD, DEFAULT_STD, DEFAULT_STD};
    normalize->AddInput("ExternalSourceOutput", "cpu");
    normalize->AddArg("mean", mean);
    normalize->AddArg("stddev", std);
    normalize->AddOutput("NormalizeOutput", "cpu");
    int ret = pipeline.Build({externalInput, normalize}, "NormalizeOutput");
    EXPECT_EQ(ret, SUCCESS);
    Tensor output;
    std::unordered_map<std::string, std::vector<Tensor>> inputs;
    inputs["ExternalSourceOutput"].push_back(inputTensor);

    // run pipeline with copy input
    ret = pipeline.Run(inputs, output, true);
    EXPECT_EQ(ret, SUCCESS);
    // compare the output with ground truth
    const float expectedValue = (UNIFORM_VALUE_FLOAT - DEFAULT_MEAN) / DEFAULT_STD;
    float* outputDataPtr = static_cast<float*>(output.Ptr());
    for (size_t i = 0; i < numElements; i++) {
        EXPECT_NEAR(outputDataPtr[i], expectedValue, 1e-5f)
            << "Mismatch at index " << i
            << ", expected: " << expectedValue
            << ", actual: " << outputDataPtr[i];
    }

    // run pipeline without copy input
    Tensor output2;
    ret = pipeline.Run(inputs, output2, false);
    EXPECT_EQ(ret, SUCCESS);
    // compare the output with ground truth
    float* outputDataPtr2 = static_cast<float*>(output2.Ptr());
    for (size_t i = 0; i < numElements; i++) {
        EXPECT_NEAR(outputDataPtr2[i], expectedValue, 1e-5f)
            << "Mismatch at index " << i
            << ", expected: " << expectedValue
            << ", actual: " << outputDataPtr2[i];
    }
}

TEST_F(PipelineTest, Test_Pipeline_Run_With_Invalid_Tensor_DataType_Should_Fail)
{
    std::vector<size_t> tensorShape = {BATCH_SIZE, HEIGHT, WIDTH, CHANNEL};
    const size_t numElements = BATCH_SIZE * HEIGHT * WIDTH * CHANNEL;
    std::vector<int8_t> inputData(numElements, UNIFORM_VALUE_INT8);
    auto inputDataPtr = static_cast<void*>(inputData.data());
    Tensor inputTensor(inputDataPtr, tensorShape, DataType::INT8, TensorFormat::NHWC, "cpu");

    Tensor output;
    std::unordered_map<std::string, std::vector<Tensor>> inputs;
    inputs["ExternalSourceOutput"].push_back(inputTensor);

    // run pipeline with copy input
    Pipeline pipeline(VALID_THREAD_NUM, true);
    auto ret = pipeline.Run(inputs, output, true);
    EXPECT_EQ(ret, ERR_ACC_DATA_PROPERTY_CONVERT_FAILURE);
}

TEST_F(PipelineTest, Test_Pipeline_Run_With_Multiple_Inputs_Should_Fail)
{
    Tensor input;
    std::unordered_map<std::string, std::vector<Tensor>> inputs;
    inputs["ExternalSourceOutput"].push_back({input});
    inputs["NormalizeOutput"].push_back({input});
    Tensor output;
    Pipeline pipeline(VALID_THREAD_NUM, true);
    auto ret = pipeline.Run(inputs, output, true);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(PipelineTest, Test_Pipeline_Run_With_Input_Vector_Size_Zero_Should_Fail)
{
    std::unordered_map<std::string, std::vector<Tensor>> inputs;
    inputs["ExternalSourceOutput"] = {};
    Tensor output;
    Pipeline pipeline(VALID_THREAD_NUM, true);
    auto ret = pipeline.Run(inputs, output, true);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}