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
 * Description: Processing of the audio.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#include "acc/audio/Audio.h"

#include <soxr.h>
#include <vector>
#include <cmath>
#include <string>
#include <future>

#include "acc/utils/AudioUtils.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
#include "acc/utils/ThreadPool.h"
#include "acc/tensor/Tensor.h"
#include "securec.h"

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

namespace {
using namespace Acc;

ErrorCode LoadAudioData(const char* path, Acc::AudioData& audioData)
{
    ErrorCode ret = Acc::AudioDecode(path, audioData);
    if (ret != SUCCESS) {
        return ret;
    }
    if (audioData.samples.empty() || audioData.numChannels <= 0 || audioData.sampleRate <= 0) {
        LogError << "Invalid audio data: empty samples, zero channels or zero sample rate."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode ProcessAudioChannels(const Acc::AudioData& audioData, std::vector<float>& monoAudio)
{
    const size_t numSamplesPerChannel = audioData.samples.size() / audioData.numChannels;
    monoAudio.resize(numSamplesPerChannel);
    ErrorCode ret = SUCCESS;

    if (audioData.numChannels == 1) {
        monoAudio = audioData.samples;
    } else {
        ret = MixChannelsInterleaved(monoAudio.data(), audioData.samples.data(), numSamplesPerChannel,
                                     audioData.numChannels);
        if (ret != SUCCESS) {
            LogError << "Failed to process audio channels" << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    return ret;
}

ErrorCode SoxrResample(const std::vector<float>& monoAudio, uint32_t originalSampleRate, int targetSampleRate,
                       Tensor& resultBuffer, size_t bufferCapacity, size_t& outputLength)
{
    soxr_quality_spec_t qSpec = soxr_quality_spec(SOXR_HQ, 0);
    soxr_io_spec_t ioSpec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);

    float* bufferPtr = static_cast<float*>(resultBuffer.Ptr());
    if (bufferPtr == nullptr) {
        LogError << "Result buffer tensor has null data pointer." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }

    soxr_error_t soxrError = soxr_oneshot(originalSampleRate, targetSampleRate, 1, monoAudio.data(), monoAudio.size(),
                                          nullptr, bufferPtr, bufferCapacity, &outputLength, &ioSpec, &qSpec, nullptr);
    if (soxrError != nullptr) {
        LogError << "Audio resampling failed: " << soxrError << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    return SUCCESS;
}

ErrorCode ResampleAudio(const std::vector<float>& monoAudio, int originalSr, std::optional<int> sr, Tensor& result,
                        size_t& expectedOutputLen, size_t& outputLength)
{
    if (monoAudio.empty()) {
        LogError << "Mono audio is empty" << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    ErrorCode ret = SUCCESS;

    if (sr.has_value()) {
        const double ratio = static_cast<double>(sr.value()) / originalSr;
        // ceil rounds up to ensure no audio data is lost
        expectedOutputLen = std::ceil(static_cast<double>(monoAudio.size()) * ratio);
        std::shared_ptr<std::vector<float>> buffer = std::make_shared<std::vector<float>>(expectedOutputLen);
        std::shared_ptr<void> dataPtr(buffer, static_cast<void*>(buffer->data()));
        result = Tensor(dataPtr, {expectedOutputLen}, DataType::FLOAT32, TensorFormat::ND, "cpu");
        // perform resampling
        ret = SoxrResample(monoAudio, originalSr, sr.value(), result, expectedOutputLen, outputLength);
    } else {
        expectedOutputLen = monoAudio.size();
        outputLength = expectedOutputLen;
        std::shared_ptr<std::vector<float>> buffer = std::make_shared<std::vector<float>>(expectedOutputLen);
        std::shared_ptr<void> dataPtr(buffer, static_cast<void*>(buffer->data()));
        result = Tensor(dataPtr, {expectedOutputLen}, DataType::FLOAT32, TensorFormat::ND, "cpu");
        // directly copy without resampling
        float* bufferPtr = static_cast<float*>(result.Ptr());
        if (bufferPtr == nullptr) {
            LogError << "Result buffer tensor has null data pointer." << GetErrorInfo(ERR_INVALID_POINTER);
            return ERR_INVALID_POINTER;
        }
        std::copy(monoAudio.begin(), monoAudio.end(), bufferPtr);
    }

    return ret;
}

ErrorCode FillRemainingBuffer(Tensor& result, size_t expectedOutputLen, size_t actualOutputLen)
{
    if (actualOutputLen >= expectedOutputLen) {
        return SUCCESS;
    }

    size_t remainBufferSize = (expectedOutputLen - actualOutputLen) * sizeof(float);
    void* resultPtr = result.Ptr();
    if (resultPtr == nullptr) {
        LogError << "Tensor data pointer is null" << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }

    char* remainingBuffer = static_cast<char*>(resultPtr) + (actualOutputLen * sizeof(float));
    errno_t result_code = memset_s(remainingBuffer, remainBufferSize, 0, remainBufferSize);
    if (result_code != 0) {
        LogError << "Memset out of range" << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }

    return SUCCESS;
}

ErrorCode ClearAudioBuffer(Tensor& buffer)
{
    if (buffer.Ptr() == nullptr) {
        LogError << "Buffer tensor data pointer is nullptr" << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }

    errno_t result_code = memset_s(buffer.Ptr(), buffer.NumBytes(), 0, buffer.NumBytes());
    if (result_code != 0) {
        LogError << "Memset_s out of range" << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }

    return SUCCESS;
}
} // namespace

namespace Acc {

ErrorCode LoadAudio(const char* path, Tensor& result, int& originalSr, std::optional<int> sr)
{
    Acc::AudioData audioData;
    ErrorCode ret = LoadAudioData(path, audioData);
    originalSr = audioData.sampleRate;
    if (ret != SUCCESS) {
        LogError << "Load audio data failed" << GetErrorInfo(ret);
        return ret;
    }

    std::vector<float> monoAudio;
    ret = ProcessAudioChannels(audioData, monoAudio);
    if (ret != SUCCESS) {
        LogError << "Process audio channels failed" << GetErrorInfo(ret);
        return ret;
    }

    size_t outputLength = 0;
    size_t expectedOutputLen = 0;

    ret = ResampleAudio(monoAudio, originalSr, sr, result, expectedOutputLen, outputLength);
    if (ret != SUCCESS) {
        LogError << "Resample audio failed" << GetErrorInfo(ret);
        if (ClearAudioBuffer(result) != SUCCESS) {
            return ERR_BAD_FREE;
        }
        return ret;
    }

    ret = FillRemainingBuffer(result, expectedOutputLen, outputLength);
    if (ret != SUCCESS) {
        LogError << "Fill remaining buffer failed" << GetErrorInfo(ret);
        if (ClearAudioBuffer(result) != SUCCESS) {
            return ERR_BAD_FREE;
        }
        return ret;
    }

    return SUCCESS;
}

ErrorCode LoadAudioSingle(const std::string path, Tensor& result, int& originalSr, std::optional<int> sr)
{
    ErrorCode ret = CheckSingleAudioInputs(path.c_str(), sr);
    if (ret != SUCCESS) {
        LogError << "Check audio inputs failed" << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    ret = LoadAudio(path.c_str(), result, originalSr, sr);
    if (ret != SUCCESS) {
        LogError << "Load audio failed" << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode LoadAudioBatch(const std::vector<std::string> wavFiles, std::vector<Tensor>& results,
                         std::vector<int>& originalSrs, std::optional<int> sr)
{
    constexpr int kMaxBatch = 128;
    if (wavFiles.size() < 1 || wavFiles.size() > kMaxBatch) {
        LogError << "Audio count must be in [1, " << kMaxBatch << "], got: " << wavFiles.size()
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    for (const auto& path : wavFiles) {
        ErrorCode ret = CheckSingleAudioInputs(path.c_str(), sr);
        if (ret != SUCCESS) {
            LogError << "Load audio failed for file: " << path << GetErrorInfo(ret);
            return ret;
        }
    }

    const int batchSize = std::min(static_cast<int>(wavFiles.size()), kMaxBatch);
    results.resize(batchSize);
    originalSrs.resize(batchSize);

    auto& pool = Acc::ThreadPool::GetInstance();
    std::atomic<bool> errorOccurred(false);
    std::vector<std::future<void>> futures(batchSize);

    for (int i = 0; i < batchSize; ++i) {
        if (errorOccurred.load()) {
            break;
        }
        const std::string wavPath = wavFiles[i];
        futures[i] = pool.Submit([wavPath, i, &results, &originalSrs, sr, &errorOccurred]() {
            ErrorCode ret = LoadAudio(wavPath.c_str(), results[i], originalSrs[i], sr);
            if (ret != SUCCESS) {
                errorOccurred.store(true);
                throw std::runtime_error("LoadAudio failed with code: " + std::to_string(ret));
            }
        });
    }

    ErrorCode ret = SUCCESS;
    for (auto& future : futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            LogError << "Thread terminated with exception: " << GetErrorInfo(ERR_INVALID_POINTER);
            ret = ERR_INVALID_POINTER;
        }
    }

    if (ret != SUCCESS) {
        results.clear();
        originalSrs.clear();
    }

    return ret;
}
} // namespace Acc