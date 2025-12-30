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
 * @Date: 2025-2-20 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-20 9:00:00
 */
#include <gtest/gtest.h>
#include <thread>
#include <unistd.h>

#include "accdata_pipeline.h"
#include "accdata_op_spec.h"
#include "logger.h"

namespace {
using namespace acclib::accdata;

const int MIN_BATCH_SIZE = 1;
const int MAX_BATCH_SIZE = 1024;
const int MIN_THREAD_NUM = 1;
const int MIN_QUEUE_DEPTH = 2;
const int MAX_QUEUE_DEPTH = 128;
const int INVALID_BATCH_SIZE = 1025;
const int INVALID_THREAD_NUM = 1025;
const int INVALID_QUEUE_DEPTH = 129;

class TestPipelineBasic : public ::testing::Test {
public:
    void SetUp() {}
    void TearDown() {}
};

class TestDataNode {
public:
    std::shared_ptr<AccDataOpSpec> spec;
    std::string outputName;
};

static uint32_t g_logicId = 0;

std::string GetOutputName(std::string prefix)
{
    return prefix + "_" + std::to_string(g_logicId++);
}

TestDataNode OpsExternalSource()
{
    auto externalInput = AccDataOpSpec::Create("ExternalSource");
    auto outputName = GetOutputName("ExternalSource");
    externalInput->AddOutput(outputName, "cpu");
    return { externalInput, outputName };
}

TestDataNode OpsToTensor(std::string inputName, TensorLayout layout)
{
    auto toTensor = AccDataOpSpec::Create("ToTensor");
    toTensor->AddInput(inputName, "cpu");
    toTensor->AddArg("layout", static_cast<int64_t>(layout));
    auto outputName = GetOutputName("ToTensor");
    toTensor->AddOutput(outputName, "cpu");
    return { toTensor, outputName };
}

TestDataNode OpsResizeCrop(std::string inputName, std::vector<int64_t> resizeInfo, std::vector<int64_t> cropInfo,
    std::string interpolationMode, std::string roundMode = "round")
{
    auto resizeCrop = AccDataOpSpec::Create("ResizeCrop");
    resizeCrop->AddInput(inputName, "cpu");
    resizeCrop->AddArg("resize", resizeInfo);
    resizeCrop->AddArg("crop", cropInfo);
    resizeCrop->AddArg("interpolation_mode", interpolationMode);
    if (roundMode != "") {
        resizeCrop->AddArg("round_mode", roundMode);
    }
    resizeCrop->AddArg("crop_pos_x", 0.5f);
    resizeCrop->AddArg("crop_pos_y", 0.5f);
    auto outputName = GetOutputName("ResizeCrop");
    resizeCrop->AddOutput(outputName, "cpu");
    return { resizeCrop, outputName };
}

TestDataNode OpsNormalize(std::string inputName, std::vector<float> mean, std::vector<float> std, float scale = 1.0)
{
    auto norm = AccDataOpSpec::Create("Normalize");
    norm->AddInput(inputName, "cpu");
    norm->AddArg("mean", mean);
    norm->AddArg("stddev", mean);
    if (scale != 1.0) {
        norm->AddArg("scale", scale);
    }
    auto outputName = GetOutputName("Normalize");
    norm->AddOutput(outputName, "cpu");
    return { norm, outputName };
}

class TestPipelineBuildNoFusion : public ::testing::Test {
public:
    void SetUp()
    {
        pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
        externalInput = OpsExternalSource();
        toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
        resize_crop = OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
        norm = OpsNormalize(resize_crop.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });
        opSpecs.push_back(externalInput.spec);
        opSpecs.push_back(toTensor.spec);
        opSpecs.push_back(resize_crop.spec);
        opSpecs.push_back(norm.spec);
    }

    void TearDown() {}

    std::shared_ptr<AccDataPipeline> pipe;
    TestDataNode externalInput;
    TestDataNode toTensor;
    TestDataNode resize_crop;
    TestDataNode norm;
    std::vector<std::shared_ptr<AccDataOpSpec>> opSpecs;
};

TEST_F(TestPipelineBuildNoFusion, BuildWithSameOpSpec)
{
    AccDataErrorCode errorCode;

    errorCode = pipe->Build({ externalInput.spec, externalInput.spec }, { toTensor.outputName });
    EXPECT_EQ(errorCode, AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestPipelineBuildNoFusion, BuildSuccessNofusion_1) // Add last node
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineBuildNoFusion, BuildSuccessNofusion_2) // Add mid node
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { resize_crop.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineBuildNoFusion, BuildSuccessNofusion_3) // Add two node
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { resize_crop.outputName, norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineBuildNoFusion, BuildSuccessNofusion_4) // Add all node
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { toTensor.outputName, resize_crop.outputName, norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineBuildNoFusion, BuildErrorNofusion_1) // Invalid spec
{
    AccDataErrorCode errCode;

    auto errSpec = AccDataOpSpec::Create("tom");
    errCode = pipe->Build({ errSpec }, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestPipelineBuildNoFusion, BuildErrorNofusion_2) // Invalid output
{
    AccDataErrorCode errCode;

    errCode = pipe->Build({ norm.spec }, { "tom" });
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestPipelineBuildNoFusion, BuildErrorNofusion_3) // Build Twice
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Build(opSpecs, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);
}

TEST_F(TestPipelineBuildNoFusion, BuildErrorNofusion_4) // Build with useless op
{
    AccDataErrorCode errCode;

    pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);

    externalInput = OpsExternalSource();
    toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
    resize_crop = OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
    norm = OpsNormalize(resize_crop.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });

    TestDataNode useLessExternalInput = OpsExternalSource();
    TestDataNode useLessToTensor = OpsToTensor(useLessExternalInput.outputName, TensorLayout::NCHW);
    TestDataNode useLessResizeCrop =
        OpsResizeCrop(useLessToTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
    TestDataNode uselessNorm = OpsNormalize(useLessResizeCrop.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });

    errCode = pipe->Build({ externalInput.spec, toTensor.spec, resize_crop.spec, norm.spec, useLessExternalInput.spec,
        useLessToTensor.spec, useLessResizeCrop.spec, uselessNorm.spec },
        { norm.outputName, uselessNorm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineBuildNoFusion, BuildErrorNofusion_5)
{
    AccDataErrorCode errCode;

    errCode = pipe->Build({}, {norm.outputName});
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_ERROR);
}

TEST_F(TestPipelineBuildNoFusion, BuildErrorNofusion_6)
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, {});
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_ERROR);
}

class TestPipelineBuildFusion : public ::testing::Test {
public:
    void SetUp()
    {
        pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, true);
        externalInput = OpsExternalSource();
        toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
        resize_crop = OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
        norm = OpsNormalize(resize_crop.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });
        opSpecs.push_back(externalInput.spec);
        opSpecs.push_back(toTensor.spec);
        opSpecs.push_back(resize_crop.spec);
        opSpecs.push_back(norm.spec);

        Logger::SetLogLevelStr("debug"); // capture debug info

        buffer.str(std::string()); // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }
    void TearDown()
    {
        Logger::SetLogLevelStr("info");
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;
    }
    std::shared_ptr<AccDataPipeline> pipe;
    TestDataNode externalInput;
    TestDataNode toTensor;
    TestDataNode resize_crop;
    TestDataNode norm;
    std::vector<std::shared_ptr<AccDataOpSpec>> opSpecs;
    std::stringstream buffer;
    std::streambuf *sbuf;
};

// L0, L1 用例
TEST_F(TestPipelineBuildFusion, BuildPipelineSuccess)
{
    auto newPipeA = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, true);
    EXPECT_NE(newPipeA.get(), nullptr);

    auto newPipeB = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    EXPECT_NE(newPipeB.get(), nullptr);

    auto maxThreadNum = sysconf(_SC_NPROCESSORS_ONLN);
    auto newPipeC = AccDataPipeline::Create(MAX_BATCH_SIZE, maxThreadNum, MAX_QUEUE_DEPTH, true);
    EXPECT_NE(newPipeC.get(), nullptr);

    auto newPipeD = AccDataPipeline::Create(MAX_BATCH_SIZE, maxThreadNum, MAX_QUEUE_DEPTH, false);
    EXPECT_NE(newPipeD.get(), nullptr);
}

// L0, L1 用例
TEST_F(TestPipelineBuildFusion, BuildPipelineWithInvalidBatchSize)
{
    auto newPipeA = AccDataPipeline::Create(0, MIN_THREAD_NUM, MIN_QUEUE_DEPTH);
    EXPECT_EQ(newPipeA.get(), nullptr);
    auto newPipeB = AccDataPipeline::Create(INVALID_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH);
    EXPECT_EQ(newPipeB.get(), nullptr);
}

// L0, L1 用例
TEST_F(TestPipelineBuildFusion, BuildPipelineWithInvalidThreadNum)
{
    auto newPipeA = AccDataPipeline::Create(MIN_BATCH_SIZE, 0, MIN_QUEUE_DEPTH);
    EXPECT_EQ(newPipeA.get(), nullptr);
    auto newPipeB = AccDataPipeline::Create(MIN_BATCH_SIZE, INVALID_THREAD_NUM, MIN_QUEUE_DEPTH);
    EXPECT_EQ(newPipeB.get(), nullptr);
}

// L0, L1 用例
TEST_F(TestPipelineBuildFusion, BuildPipelineWithInvalidQueueDepth)
{
    auto newPipeA = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, 0);
    EXPECT_EQ(newPipeA.get(), nullptr);
    auto newPipeB = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, INVALID_QUEUE_DEPTH);
    EXPECT_EQ(newPipeB.get(), nullptr);
}

TEST_F(TestPipelineBuildFusion, BuildSuccessFusion) // Fusion success
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("ToTensorResizeCropNormalize"), std::string::npos);
}

TEST_F(TestPipelineBuildFusion, BuildSuccessNoFusion_1) // Fusion fail
{
    AccDataErrorCode errCode;
    pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);

    errCode = pipe->Build(opSpecs, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    auto logger_string = buffer.str();
    EXPECT_EQ(logger_string.find("ToTensorResizeCropNormalize"), std::string::npos);
}

TEST_F(TestPipelineBuildFusion, BuildSuccessNoFusion_2) // Fusion Fail
{
    AccDataErrorCode errCode;

    errCode = pipe->Build(opSpecs, { resize_crop.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    auto logger_string = buffer.str();
    EXPECT_EQ(logger_string.find("ToTensorResizeCropNormalize"), std::string::npos);
}

TEST_F(TestPipelineBuildFusion, BuildSuccessNoFusion_3) // Fusion Fail
{
    AccDataErrorCode errCode;

    pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, true);
    externalInput = OpsExternalSource();
    resize_crop = OpsResizeCrop(externalInput.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
    norm = OpsNormalize(resize_crop.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });
    errCode = pipe->Build({ externalInput.spec, resize_crop.spec, norm.spec }, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    auto logger_string = buffer.str();
    EXPECT_EQ(logger_string.find("ToTensorResizeCropNormalize"), std::string::npos);
}

TEST_F(TestPipelineBuildFusion, BuildSuccessNoFusion_4) // Fusion Fail
{
    AccDataErrorCode errCode;

    pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, true);
    externalInput = OpsExternalSource();
    toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
    norm = OpsNormalize(toTensor.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });
    resize_crop = OpsResizeCrop(norm.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
    errCode =
        pipe->Build({ externalInput.spec, toTensor.spec, norm.spec, resize_crop.spec }, { resize_crop.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    auto logger_string = buffer.str();
    EXPECT_EQ(logger_string.find("ToTensorResizeCropNormalize"), std::string::npos);
}

class TestPipelineRun : public ::testing::Test {
public:
    void SetUp()
    {
        buffer.str(std::string()); // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }

    void TearDown()
    {
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;
    }

    template <typename T>
    std::shared_ptr<AccDataTensorList> SetupInputTensorList(TensorShape tensorShape, TensorLayout layout)
    {
        auto inputTensor = AccDataTensorList::Create(1);
        size_t tensorSize = tensorShape[1] * tensorShape[2] * tensorShape[3]; // c/h/w: 1, 2, 3
        std::vector<T> datas(tensorSize);

        if (std::is_same<T, uint8_t>::value) {
            inputTensor->operator[](0).Copy(datas.data(), tensorShape, TensorDataType::UINT8);
        } else if (std::is_same<T, float>::value) {
            inputTensor->operator[](0).Copy(datas.data(), tensorShape, TensorDataType::FP32);
        }

        inputTensor->operator[](0).SetLayout(layout);

        return inputTensor;
    }

    void RunWithCheck(std::shared_ptr<AccDataPipeline> pipe, const std::string &inputName,
        std::shared_ptr<AccDataTensorList> input, int outputCnt)
    {
        AccDataErrorCode errCode = AccDataErrorCode::H_OK;
        std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> inputMap;
        inputMap[inputName] = input;

        std::vector<std::shared_ptr<AccDataTensorList>> outputs;
        errCode = pipe->Run(inputMap, outputs, false);
        EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
        EXPECT_EQ(outputs.size(), outputCnt);
        for (int i = 0; i < outputCnt; i++) {
            EXPECT_EQ(outputs[i]->NumTensors(), 1);
        }

        errCode = pipe->Run(inputMap, outputs, true);
        EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
        EXPECT_EQ(outputs.size(), outputCnt);
        for (int i = 0; i < outputCnt; i++) {
            EXPECT_EQ(outputs[i]->NumTensors(), 1);
        }
    }

    void FunctionWithTimeout(std::function<void()> func, std::atomic<bool> &okSign)
    {
        std::thread runThread(func);
        int waitCount = 10; // waitCount=10 is 100 ms
        while (!okSign && waitCount >= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // sleep 10 ms
            waitCount -= 1;
        }
        EXPECT_LT(waitCount, 0);
        EXPECT_EQ(okSign, false);
        pthread_cancel(runThread.native_handle());
        runThread.join();
    }

    std::stringstream buffer;
    std::streambuf *sbuf;
    std::shared_ptr<AccDataTensorList> input_tensorlist_float_1;
    std::shared_ptr<AccDataTensorList> input_tensorlist_float_2;
};

TEST_F(TestPipelineRun, FeedInputSuccess) // FeedInputSuccess
{
    AccDataErrorCode errCode;
    auto inputTensor = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;

    errCode = pipe->Run({ { externalInput.outputName, inputTensor } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run({ { externalInput.outputName, inputTensor } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineRun, FeedInputNotExternalSourece) // FeedInputNotExternalSourece
{
    AccDataErrorCode errCode;
    auto inputTensor = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NHWC);
    auto norm = OpsNormalize(toTensor.outputName, {0.1, 0.1, 0.1}, {0.2, 0.2, 0.2});
    errCode = pipe->Build({ externalInput.spec, toTensor.spec, norm.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;

    errCode = pipe->Run({ { toTensor.outputName, inputTensor } }, outputs, false);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineRun, FeedInputSameSuccess) // FeedInputSame
{
    AccDataErrorCode errCode;
    auto inputTensor_1 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);
    auto inputTensor_2 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> inputMap;
    inputMap[externalInput.outputName] = inputTensor_1;
    inputMap[externalInput.outputName] = inputTensor_2;

    std::vector<std::shared_ptr<AccDataTensorList>> opOutputs;
    errCode = pipe->Run(inputMap, opOutputs, false); // Run no copy first time
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run(inputMap, opOutputs, false); // Run no copy second time
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run(inputMap, opOutputs, true); // Run no copy first time
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run(inputMap, opOutputs, true); // Run no copy second time
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineRun, RunOpNotExistWithLogCheck) // RunOpNotExistWithLogCheck
{
    AccDataErrorCode errCode;
    auto input_tensor_uint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run({ { "tom", input_tensor_uint8 } }, outputs, false);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run({ { "tom", input_tensor_uint8 } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);

    auto logger_string = buffer.str();
}

TEST_F(TestPipelineRun, RunOpNotExist) // RunOpNotExist
{
    AccDataErrorCode errCode;
    auto input_tensor_uint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run({ { "Tom", input_tensor_uint8 } }, outputs, false);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run({ { "Tom", input_tensor_uint8 } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);
}

TEST_F(TestPipelineRun, RunCheckFusion) // RunCheckFusion
{
    AccDataErrorCode errCode;
    auto input_tensor_uint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
    auto resizeCrop = OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear");
    auto norm = OpsNormalize(resizeCrop.outputName, { 0.5f, 0.5f, 0.5f }, { 0.4f, 0.4f, 0.5f });

    errCode = pipe->Build({ externalInput.spec, toTensor.spec, resizeCrop.spec, norm.spec }, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    RunWithCheck(pipe, externalInput.outputName, input_tensor_uint8, 1);

    pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, true);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec, resizeCrop.spec, norm.spec }, { norm.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    RunWithCheck(pipe, externalInput.outputName, input_tensor_uint8, 1);
}

TEST_F(TestPipelineRun, RunTwoOp) // RunTwoOp a->b a->c, build({b, c}) -> 不支持
{
    AccDataErrorCode errCode;
    auto input_tensor_uint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput1 = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput1.outputName, TensorLayout::NCHW);
    auto externalInput2 = OpsExternalSource();
    auto toTensor2 = OpsToTensor(externalInput2.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ toTensor.spec, toTensor2.spec, externalInput1.spec, externalInput2.spec },
        { toTensor.outputName, toTensor2.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run(
        { { externalInput1.outputName, input_tensor_uint8 }, { externalInput2.outputName, input_tensor_uint8 } },
        outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run(
        { { externalInput1.outputName, input_tensor_uint8 }, { externalInput2.outputName, input_tensor_uint8 } },
        outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    EXPECT_EQ(outputs.size(), 2);
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(outputs[i]->NumTensors(), 1);
    }
}

TEST_F(TestPipelineRun, RunWithoutBuild) // RunWithoutBuild
{
    AccDataErrorCode errCode;
    auto inputTensorUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run({ { externalInput.outputName, inputTensorUint8 } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);

    errCode = pipe->Run({ { externalInput.outputName, inputTensorUint8 } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);
}

TEST_F(TestPipelineRun, RunErrorTwice) // RunErrorTwice
{
    AccDataErrorCode errCode;
    auto inputTensorUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run({}, outputs, false);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run({}, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);

    errCode = pipe->Run({}, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);

    errCode = pipe->Run({}, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_PIPELINE_STATE_ERROR);
}

TEST_F(TestPipelineRun, RunLessThanDepth) // RunLessThanDepth
{
    AccDataErrorCode errCode;
    auto inputTensorUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run({ { externalInput.outputName, inputTensorUint8 } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
    errCode = pipe->Run({ { externalInput.outputName, inputTensorUint8 } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineRun, RunWithOutput) // RunWithOutput
{
    AccDataErrorCode errCode;
    auto inputTensorUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    for (int i = 0; i < 100; i++) { // run 100 count
        RunWithCheck(pipe, externalInput.outputName, inputTensorUint8, 1);
    }
}

TEST_F(TestPipelineRun, GetoutputRunError) // GetoutputRunError
{
    AccDataErrorCode errCode;
    auto inputTensorUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);

    auto pipe = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipe->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipe->Run({ {} }, outputs, false);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);

    errCode = pipe->Run({ {} }, outputs, true);
    EXPECT_NE(errCode, AccDataErrorCode::H_OK);
}

TEST_F(TestPipelineRun, RunWithNullpter)
{
    AccDataErrorCode errCode;
    auto pipeA = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);

    errCode = pipeA->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipeA->Run({ { externalInput.outputName, nullptr } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeA->Run({ { externalInput.outputName, nullptr } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeA->Run({ { externalInput.outputName, NULL } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeA->Run({ { externalInput.outputName, NULL } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeA->Run({ { externalInput.outputName, 0 } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeA->Run({ { externalInput.outputName, 0 } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);

    auto pipeB = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, true);

    errCode = pipeB->Build({ externalInput.spec, toTensor.spec }, { toTensor.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipeB->Run({ { externalInput.outputName, nullptr } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeB->Run({ { externalInput.outputName, nullptr } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeB->Run({ { externalInput.outputName, NULL } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeB->Run({ { externalInput.outputName, NULL } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeB->Run({ { externalInput.outputName, 0 } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
    errCode = pipeB->Run({ { externalInput.outputName, 0 } }, outputs, true);
    EXPECT_EQ(errCode, AccDataErrorCode::H_COMMON_NULLPTR);
}

TEST_F(TestPipelineRun, SetRoundModeSuccessfully)
{
    Logger::SetLogLevelStr("debug");

    AccDataErrorCode errCode;
    auto inputTensor = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, TensorLayout::NHWC);
    // set round mode truncate
    auto pipeA = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    auto externalInput = OpsExternalSource();
    auto toTensor = OpsToTensor(externalInput.outputName, TensorLayout::NCHW);
    auto resize_crop =
        OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear", "truncate");

    errCode = pipeA->Build({ externalInput.spec, toTensor.spec, resize_crop.spec }, { resize_crop.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    std::vector<std::shared_ptr<AccDataTensorList>> outputs;
    errCode = pipeA->Run({ { externalInput.outputName, inputTensor } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Crop args: PosX = 0.5, PosY = 0.5, W = 1024, H = 1024, round mode = 'truncate'."),
        std::string::npos);
    // set round mode round
    auto pipeB = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    resize_crop = OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear", "round");

    errCode = pipeB->Build({ externalInput.spec, toTensor.spec, resize_crop.spec }, { resize_crop.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipeB->Run({ { externalInput.outputName, inputTensor } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Crop args: PosX = 0.5, PosY = 0.5, W = 1024, H = 1024, round mode = 'round'."),
        std::string::npos);
    // use default round mode
    auto pipeC = AccDataPipeline::Create(MIN_BATCH_SIZE, MIN_THREAD_NUM, MIN_QUEUE_DEPTH, false);
    resize_crop = OpsResizeCrop(toTensor.outputName, { 1920LL, 1080LL }, { 1024LL, 1024LL }, "bilinear", "");

    errCode = pipeC->Build({ externalInput.spec, toTensor.spec, resize_crop.spec }, { resize_crop.outputName });
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    errCode = pipeC->Run({ { externalInput.outputName, inputTensor } }, outputs, false);
    EXPECT_EQ(errCode, AccDataErrorCode::H_OK);

    logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Crop args: PosX = 0.5, PosY = 0.5, W = 1024, H = 1024, round mode = 'round'."),
        std::string::npos);
    Logger::SetLogLevelStr("info");
}
} // namespace
