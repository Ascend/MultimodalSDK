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
 * Description: Internal audio utils header file.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H
#include <cstddef>
#include <vector>
#include <optional>
#include "acc/ErrorCode.h"

namespace Acc {

struct AudioData {
    std::vector<float> samples;
    uint32_t sampleRate;
    uint16_t numChannels;
    uint16_t bitsPerSample;
    uint32_t numSamples;
};

/**
 * @brief Checks validity of audio input parameters.
 * @param path Path to the audio file.
 * @param sr Optional target sample rate.
 * @return ErrorCode
 */
ErrorCode CheckSingleAudioInputs(const char* path, std::optional<int> sr = std::nullopt);

/**
 * @brief Mixes interleaved multi-channel audio into mono.
 * @param output Output mono buffer.
 * @param input Input interleaved audio buffer.
 * @param numFrames Number of audio frames.
 * @param numChannels Number of input channels.
 * @return ErrorCode
 */
ErrorCode MixChannelsInterleaved(float* output, const float* input, size_t numFrames, int numChannels);

/**
 * @brief Decodes a WAV audio file and converts the data to floating-point PCM.
 * @param filePath Path to the input WAV audio file.
 * @param outputAudioData Output structure to store the decoded audio data.
 * @return ErrorCode
 */
ErrorCode AudioDecode(const char* filePath, AudioData& outputAudioData);
} // namespace Acc
#endif