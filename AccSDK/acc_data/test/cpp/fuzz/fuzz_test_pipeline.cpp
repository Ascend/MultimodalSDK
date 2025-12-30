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
 * @Date: 2025-4-1 19:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-4-1 19:00:00
 */

#include <gtest/gtest.h>

#include "secodeFuzz.h"

#include "interface/accdata_pipeline.h"
#include "interface/accdata_error_code.h"

#include "random_utils.h"

namespace acclib {
namespace accdata {
struct TestDataNode {
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

TestDataNode OpsToTensor(std::string opName, std::string inputName, TensorLayout layout)
{
    auto toTensor = AccDataOpSpec::Create(opName);
    toTensor->AddInput(inputName, "cpu");
    toTensor->AddArg("layout", static_cast<int64_t>(layout));
    auto outputName = GetOutputName("ToTensor");
    toTensor->AddOutput(outputName, "cpu");
    return { toTensor, outputName };
}

TestDataNode OpsResizeCrop(std::string opName, std::string inputName, std::vector<int64_t> resizeInfo,
    std::vector<int64_t> cropInfo, std::string interpolationMode, std::string roundMode, float cropX, float cropY)
{
    auto resizeCrop = AccDataOpSpec::Create(opName);
    resizeCrop->AddInput(inputName, "cpu");
    resizeCrop->AddArg("resize", resizeInfo);
    resizeCrop->AddArg("crop", cropInfo);
    resizeCrop->AddArg("interpolation_mode", interpolationMode);
    resizeCrop->AddArg("roundMode", roundMode);
    resizeCrop->AddArg("crop_pos_x", cropX);
    resizeCrop->AddArg("crop_pos_y", cropY);
    auto outputName = GetOutputName("ResizeCrop");
    resizeCrop->AddOutput(outputName, "cpu");
    return { resizeCrop, outputName };
}

TestDataNode OpsNormalize(std::string opName, std::string inputName, std::vector<float> mean,
    std::vector<float> std, float scale = 1.0)
{
    auto norm = AccDataOpSpec::Create(opName);
    norm->AddInput(inputName, "cpu");
    norm->AddArg("mean", mean);
    norm->AddArg("stddev", std);
    if (scale != 1.0) {
        norm->AddArg("scale", scale);
    }
    auto outputName = GetOutputName("Normalize");
    norm->AddOutput(outputName, "cpu");
    return { norm, outputName };
}

TestDataNode OpsFusion(std::string opName, std::string inputName, std::vector<int64_t> resizeInfo,
    std::vector<int64_t> cropInfo, std::string interpolationMode, std::string roundMode, float cropPosW, float cropPosH,
    std::vector<float> mean, std::vector<float> stddev, float scale, TensorLayout layout, bool withOutput = true)
{
    auto fusedOp = AccDataOpSpec::Create(opName);
    fusedOp->AddInput(inputName, "cpu");
    fusedOp->AddArg<std::vector<int64_t>>("resize", resizeInfo);
    fusedOp->AddArg<std::vector<int64_t>>("crop", cropInfo);
    fusedOp->AddArg<std::string>("interpolation_mode", interpolationMode);
    fusedOp->AddArg<std::string>("round_mode", roundMode);
    fusedOp->AddArg<float>("crop_pos_x", cropPosW);
    fusedOp->AddArg<float>("crop_pos_y", cropPosH);
    fusedOp->AddArg<std::vector<float>>("mean", mean);
    fusedOp->AddArg<std::vector<float>>("stddev", stddev);
    fusedOp->AddArg<float>("scale", scale);
    fusedOp->AddArg<int64_t>("layout", static_cast<int64_t>(layout));
    auto outputName = GetOutputName("ToTensorResizeCropNormalize");
    if (withOutput) {
        fusedOp->AddOutput(outputName, "cpu");
    }
    return { fusedOp, outputName };
}

TestDataNode OpsQWenFusion(std::string opName, std::string inputName, std::vector<float> mean,
    std::vector<float> std, int64_t minPixels, int64_t maxPixels, int64_t patchSize, int64_t temporalPatchSize,
    int64_t mergeSize, TensorLayout layout, bool withOutput = true)
{
    auto qWen = AccDataOpSpec::Create(opName);
    qWen->AddInput(inputName, "cpu");
    qWen->AddArg<std::vector<float>>("mean", mean);
    qWen->AddArg<std::vector<float>>("stddev", std);
    qWen->AddArg<int64_t>("min_pixels", minPixels);
    qWen->AddArg<int64_t>("max_pixels", maxPixels);
    qWen->AddArg<int64_t>("patch_size", patchSize);
    qWen->AddArg<int64_t>("temporal_patch_size", temporalPatchSize);
    qWen->AddArg<int64_t>("merge_size", mergeSize);
    qWen->AddArg<int64_t>("layout", static_cast<int64_t>(layout));
    auto outputName = GetOutputName("QwenFusionOp");
    if (withOutput) {
        qWen->AddOutput(outputName, "cpu");
    }
    return { qWen, outputName };
}

class FuzzTestPipeline : public testing::Test {
public:
    void SetUp()
    {
        Logger::SetLogLevelStr("error");
        DT_Set_Running_Time_Second(EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
    }

    template <typename T>
    std::shared_ptr<AccDataTensorList> SetupInputTensorList(TensorShape tensorShape, TensorLayout layout)
    {
        auto inputTensor = AccDataTensorList::Create(1);
        size_t tensorSize = tensorShape[1] * tensorShape[2] * tensorShape[3];
        std::vector<T> data(tensorSize);

        if (std::is_same<T, uint8_t>::value) {
            inputTensor->operator[](0).Copy(data.data(), tensorShape, TensorDataType::UINT8);
        } else if (std::is_same<T, float>::value) {
            inputTensor->operator[](0).Copy(data.data(), tensorShape, TensorDataType::FP32);
        }

        inputTensor->operator[](0).SetLayout(layout);

        return inputTensor;
    }

    std::vector<TensorLayout> allLayoutOptions = { TensorLayout::NCHW, TensorLayout::NHWC, TensorLayout::PLAIN,
        TensorLayout::LAST };
    std::vector<TensorLayout> someLayoutOptions = { TensorLayout::NCHW, TensorLayout::NHWC };
    std::vector<std::string> roundModeOptions = { "truncate", "round", "wrong" };
    std::vector<std::string> allInterpolationModeOptions = { "bilinear", "bicubic", "wrong" };
    std::vector<std::string> someInterpolationModeOptions = { "bilinear", "wrong" };
    std::vector<std::vector<int64_t>> resizeOptions = { { 0, 0 }, { 1920, 1080 }, { 9000, 9000 } };
    std::vector<std::vector<int64_t>> cropOptions = { { 0, 0 }, { 1024, 1024 }, { 2000, 2000 }, { 9000, 9000 } };
};

TEST_F(FuzzTestPipeline, CreatePipeline)
{
    std::string caseName = "AccDataPipeline::Create";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        int batchSize = GenerateOneData<int>(0, 2000);
        int threadNum = GenerateOneData<int>(0, 200);
        int queueDepth = GenerateOneData<int>(0, 200);
        bool enableFusion = GenerateOneData<bool>(0, 1);
        (void)AccDataPipeline::Create(batchSize, threadNum, queueDepth, enableFusion);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestPipeline, BuildPipeline)
{
    std::string caseName = "AccDataPipeline::Build";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto pipe = AccDataPipeline::Create();
        auto externalInput = OpsExternalSource();
        std::vector<std::shared_ptr<AccDataOpSpec>> specs = { externalInput.spec, externalInput.spec };
        std::vector<std::string> outputs = { externalInput.outputName, externalInput.outputName };
        uint32_t specSize = GenerateOneData<uint32_t>(0, 2);
        specs.resize(specSize);
        uint32_t outputSize = GenerateOneData<uint32_t>(0, 2);
        outputs.resize(outputSize);
        uint32_t buildTime = GenerateOneData<uint32_t>(1, 2);
        for (uint32_t i = 0; i < buildTime; ++i) {
            pipe->Build(specs, outputs);
        }
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestPipeline, RunToTensorOp)
{
    std::string caseName = "AccDataPipeline::Run(to_tensor)";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto pipe = AccDataPipeline::Create();
        auto externalInput = OpsExternalSource();
        // random tensor layout
        TensorLayout outputLayout = RandomSelectOne(allLayoutOptions);
        // random op name
        std::vector<std::string> opNames = { "ToTensor", "WrongName" };
        std::string opName = RandomSelectOne(opNames);
        auto toTensor = OpsToTensor(opName, externalInput.outputName, outputLayout);
        // random input tensor
        TensorLayout inputLayout = RandomSelectOne(allLayoutOptions);
        auto inputDataUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, inputLayout);
        auto inputDataFloat = SetupInputTensorList<float>({ 1, 1920, 1080, 3 }, inputLayout);
        std::vector<std::shared_ptr<AccDataTensorList>> inputDataOptions = { inputDataUint8, inputDataFloat };
        auto inputData = RandomSelectOne(inputDataOptions);
        // build
        std::vector<std::shared_ptr<AccDataOpSpec>> specs = { externalInput.spec, toTensor.spec };
        std::vector<std::string> outputs = { toTensor.outputName };
        pipe->Build(specs, outputs);
        // run
        std::vector<std::shared_ptr<AccDataTensorList>> outputData;
        pipe->Run({ {externalInput.outputName, inputData } }, outputData, false);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestPipeline, RunResizeCropOP)
{
    std::string caseName = "AccDataPipeline::Run(resize_crop)";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto pipe = AccDataPipeline::Create();
        auto externalInput = OpsExternalSource();
        // random op name
        std::vector<std::string> opNames = { "ResizeCrop", "WrongName" };
        std::string opName = RandomSelectOne(opNames);
        // random resize and crop size
        auto resizeInfo = RandomSelectOne(resizeOptions);
        auto cropInfo = RandomSelectOne(cropOptions);
        // random round mode
        auto roundMode = RandomSelectOne(roundModeOptions);
        // random interpolation mode
        auto interpolationMode = RandomSelectOne(allInterpolationModeOptions);
        // random cropX and cropY
        float cropX = GenerateOneData<float>(-1, 1);
        float cropY = GenerateOneData<float>(-1, 1);
        auto resizeCrop = OpsResizeCrop(opName, externalInput.outputName, resizeInfo, cropInfo, interpolationMode,
            roundMode, cropX, cropY);
        // random input tensor
        TensorLayout layout = RandomSelectOne(someLayoutOptions);
        auto inputDataUint8 = SetupInputTensorList<uint8_t>({ 1, 3, 1920, 1080 }, layout);
        auto inputDataFloat = SetupInputTensorList<float>({ 1, 3, 1920, 1080 }, layout);
        std::vector<std::shared_ptr<AccDataTensorList>> inputDataOptions = { inputDataUint8, inputDataFloat };
        auto inputData = RandomSelectOne(inputDataOptions);
        // build
        std::vector<std::shared_ptr<AccDataOpSpec>> specs = { externalInput.spec, resizeCrop.spec };
        std::vector<std::string> outputs = { resizeCrop.outputName };
        pipe->Build(specs, outputs);
        // run
        std::vector<std::shared_ptr<AccDataTensorList>> outputData;
        pipe->Run({ {externalInput.outputName, inputData } }, outputData, false);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestPipeline, RunNormalizeOP)
{
    std::string caseName = "AccDataPipeline::Run(normalize)";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto pipe = AccDataPipeline::Create();
        auto externalInput = OpsExternalSource();
        // random op name
        std::vector<std::string> opNames = { "Normalize", "WrongName" };
        std::string opName = RandomSelectOne(opNames);
        // random mean and stddev
        uint32_t meanSize = GenerateOneData<uint32_t>(1, 3);
        std::vector<float> mean;
        for (uint32_t i = 0; i < meanSize; ++i) {
            mean.emplace_back(GenerateOneData<float>(-1, 1));
        }
        uint32_t stdSize = GenerateOneData<uint32_t>(1, 3);
        std::vector<float> stddev;
        for (uint32_t i = 0; i < stdSize; ++i) {
            stddev.emplace_back(GenerateOneData<float>(-1, 1));
        }
        // random scale
        float scale = GenerateOneData<float>(0, 2);
        auto norm = OpsNormalize(opName, externalInput.outputName, mean, stddev, scale);
        // random input tensor
        TensorLayout layout = RandomSelectOne(allLayoutOptions);
        auto inputDataUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, layout);
        auto inputDataFloat = SetupInputTensorList<float>({ 1, 1920, 1080, 3 }, layout);
        std::vector<std::shared_ptr<AccDataTensorList>> inputDataOptions = { inputDataUint8, inputDataFloat };
        auto inputData = RandomSelectOne(inputDataOptions);
        // build
        std::vector<std::shared_ptr<AccDataOpSpec>> specs = { externalInput.spec, norm.spec };
        std::vector<std::string> outputs = { norm.outputName };
        pipe->Build(specs, outputs);
        // run
        std::vector<std::shared_ptr<AccDataTensorList>> outputData;
        pipe->Run({ {externalInput.outputName, inputData } }, outputData, false);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestPipeline, RunFusedOp)
{
    std::string caseName = "AccDataPipeline::Run(to_tensor_resize_crop_normalize)";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto pipe = AccDataPipeline::Create();
        auto externalInput = OpsExternalSource();
        // random op name
        std::vector<std::string> opNames = { "ToTensorResizeCropNormalize", "WrongName" };
        std::string opName = RandomSelectOne(opNames);
        // random resize and crop info
        auto resizeInfo = RandomSelectOne(resizeOptions);
        auto cropInfo = RandomSelectOne(cropOptions);
        // random round mode
        auto roundMode = RandomSelectOne(roundModeOptions);
        // random interpolation mode
        auto interpolationMode = RandomSelectOne(someInterpolationModeOptions);
        // random cropX and cropY
        float cropX = GenerateOneData<float>(-1, 1);
        float cropY = GenerateOneData<float>(-1, 1);
        // random mean and stddev
        uint32_t meanSize = GenerateOneData<uint32_t>(1, 3);
        std::vector<float> mean;
        for (uint32_t i = 0; i < meanSize; ++i) {
            mean.emplace_back(GenerateOneData<float>(-1, 1));
        }
        uint32_t stdSize = GenerateOneData<uint32_t>(1, 3);
        std::vector<float> stddev;
        for (uint32_t i = 0; i < stdSize; ++i) {
            stddev.emplace_back(GenerateOneData<float>(-1, 1));
        }
        // random scale
        float scale = GenerateOneData<float>(0, 2);
        // random tensor layout
        TensorLayout outputLayout = RandomSelectOne(someLayoutOptions);
        auto fusedOp = OpsFusion(opName, externalInput.outputName, resizeInfo, cropInfo, interpolationMode, roundMode,
            cropX, cropY, mean, stddev, scale, outputLayout);
        // random input tensor
        TensorLayout inputLayout = RandomSelectOne(someLayoutOptions);
        auto inputDataUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, inputLayout);
        auto inputDataFloat = SetupInputTensorList<float>({ 1, 1920, 1080, 3 }, inputLayout);
        std::vector<std::shared_ptr<AccDataTensorList>> inputDataOptions = { inputDataUint8, inputDataFloat };
        auto inputData = RandomSelectOne(inputDataOptions);
        // build
        std::vector<std::shared_ptr<AccDataOpSpec>> specs = { externalInput.spec, fusedOp.spec };
        std::vector<std::string> outputs = { fusedOp.outputName };
        pipe->Build(specs, outputs);
        // run
        std::vector<std::shared_ptr<AccDataTensorList>> outputData;
        pipe->Run({ {externalInput.outputName, inputData } }, outputData, false);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestPipeline, RunQwenFuseddOp)
{
    std::string caseName = "AccDataPipeline::Run(qwen_fusion_op)";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto pipe = AccDataPipeline::Create();
        auto externalInput = OpsExternalSource();
        // random op name
        std::vector<std::string> opNames = { "QwenFusionOp", "WrongName" };
        std::string opName = RandomSelectOne(opNames);
        // random max and min pixels
        std::vector<int64_t> maxPixelsOptions = { 0, 28 * 28 * 1280, 9000 * 9000 };
        int64_t maxPixels = RandomSelectOne(maxPixelsOptions);
        std::vector<int64_t> minPixelsOptions = { 0, 56 * 56, 1000 * 1000, 9000 * 9000 };
        int64_t minPixels = RandomSelectOne(minPixelsOptions);
        // random patch size and merge size and temporal patch size
        std::vector<int64_t> patchSizeOptions = { -1, 14, 9000 };
        int64_t patchSize = RandomSelectOne(patchSizeOptions);
        std::vector<int64_t> mergeSizeOptions = { -1, 2, 9000 };
        int64_t mergeSize = RandomSelectOne(mergeSizeOptions);
        std::vector<int64_t> temporalPatchSizeOptions = { 1, 2 };
        int64_t temporalPatchSize = RandomSelectOne(temporalPatchSizeOptions);
        // random mean and stddev
        std::vector<float> mean;
        for (uint32_t i = 0; i < 3U; ++i) {
            mean.emplace_back(GenerateOneData<float>(0, 1));
        }
        std::vector<float> stddev;
        for (uint32_t i = 0; i < 3U; ++i) {
            stddev.emplace_back(GenerateOneData<float>(0, 1));
        }
        // random layout
        TensorLayout layout = RandomSelectOne(someLayoutOptions);
        // build
        auto qwenOp = OpsQWenFusion(opName, externalInput.outputName, mean, stddev, minPixels, maxPixels, patchSize,
            temporalPatchSize, mergeSize, layout);
        pipe->Build({ externalInput.spec, qwenOp.spec }, { qwenOp.outputName });
        // random input tensor
        TensorLayout inputLayout = RandomSelectOne(someLayoutOptions);
        auto inputDataUint8 = SetupInputTensorList<uint8_t>({ 1, 1920, 1080, 3 }, inputLayout);
        auto inputDataFloat = SetupInputTensorList<float>({ 1, 1920, 1080, 3 }, inputLayout);
        std::vector<std::shared_ptr<AccDataTensorList>> inputDataOptions = { inputDataUint8, inputDataFloat };
        auto inputData = RandomSelectOne(inputDataOptions);
        // run
        std::vector<std::shared_ptr<AccDataTensorList>> outputs;
        pipe->Run({ {externalInput.outputName, inputData} }, outputs, false);
    }
    DT_FUZZ_END()
}
}
}