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
 * Description: Audio utils file.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#include "acc/utils/AudioUtils.h"

#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <algorithm>
#include <cstring>

#include "securec.h"
#include "acc/ErrorCode.h"
#include "acc/utils/FileUtils.h"
#include "acc/utils/ErrorCodeUtils.h"
#include "acc/utils/LogImpl.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace {
using namespace Acc;
using ConvertFunc = std::function<ErrorCode(const uint8_t*, float*, size_t)>;

const size_t WAVE_FORMAT_PCM = 1;
const size_t WAVE_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t BITS_PER_SAMPLE_16 = 16;
constexpr uint16_t BITS_PER_SAMPLE_24 = 24;
constexpr uint16_t BITS_PER_SAMPLE_32 = 32;
constexpr size_t NEON_VECTOR_BITS = 128;
constexpr size_t FLOAT32_BITS = 32;
constexpr size_t NEON_FLOAT32_LANES = NEON_VECTOR_BITS / FLOAT32_BITS;
constexpr float PCM16_MAX_ABS_VALUE = 32768.0f;      // 2^15
constexpr float PCM24_MAX_ABS_VALUE = 8388608.0f;    // 2^23
constexpr float PCM32_MAX_ABS_VALUE = 2147483648.0f; // 2^31
constexpr int64_t AUDIO_MAX_FILE_SIZE = 1024 * 1024 * 50;

struct WavFmt {
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

struct AudioConvertEntry {
    uint16_t audioFormat;
    uint16_t bitsPerSample;
    ConvertFunc convert;
};

#ifdef __ARM_NEON
void ConvertPcm16ToFloatNeon(const uint8_t* raw, float* samples, size_t numSamples)
{
    const int16_t* pcmData = reinterpret_cast<const int16_t*>(raw);
    constexpr float scale = 1.0f / PCM16_MAX_ABS_VALUE;
    const float32x4_t scaleVec = vdupq_n_f32(scale);
    const float32x4_t oneVec = vdupq_n_f32(1.0f);
    const float32x4_t minusOneVec = vdupq_n_f32(-1.0f);
    size_t i = 0;
    const size_t neonSamples = numSamples - (numSamples % NEON_FLOAT32_LANES);

    for (; i < neonSamples; i += NEON_FLOAT32_LANES) {
        int16x4_t pcmVec = vld1_s16(pcmData + i);
        int32x4_t pcm32Vec = vmovl_s16(pcmVec);
        float32x4_t floatVec = vcvtq_f32_s32(pcm32Vec);
        floatVec = vmulq_f32(floatVec, scaleVec);
        floatVec = vminq_f32(vmaxq_f32(floatVec, minusOneVec), oneVec);
        vst1q_f32(samples + i, floatVec);
    }
}
#else
void ConvertPcm16ToFloatScalar(const uint8_t* raw, std::vector<float>& samples, size_t numSamples)
{
    constexpr size_t bytesPerSample = 16 / 8;
    constexpr float scale = 1.0f / PCM16_MAX_ABS_VALUE;

    for (size_t i = 0; i < numSamples; i++) {
        const size_t offset = i * bytesPerSample;

        int16_t sample = static_cast<int16_t>(raw[offset] | (raw[offset + 1] << 8));

        float value = static_cast<float>(sample) * scale;
        samples[i] = std::clamp(value, -1.0f, 1.0f);
    }
}
#endif

void ConvertPcm24ToFloatScalar(const uint8_t* raw, float* samples, size_t numSamples)
{
    constexpr size_t bytesPerSample = 24 / 8;
    constexpr int32_t signBit = 0x00800000;
    constexpr int32_t signExtMask = 0xFF000000;
    constexpr float scale = 1.0f / PCM24_MAX_ABS_VALUE;

    for (size_t i = 0; i < numSamples; ++i) {
        const size_t offset = i * bytesPerSample;

        int32_t sample = static_cast<int32_t>(raw[offset]) | (static_cast<int32_t>(raw[offset + 1]) << 8) |
                         (static_cast<int32_t>(raw[offset + 2]) << 16);

        if (sample & signBit) {
            sample |= signExtMask;
        }

        float value = static_cast<float>(sample) * scale;
        samples[i] = std::clamp(value, -1.0f, 1.0f);
    }
}

#ifdef __ARM_NEON
void ConvertPcm32ToFloatNeon(const uint8_t* raw, float* samples, size_t numSamples)
{
    const int32_t* pcmData = reinterpret_cast<const int32_t*>(raw);
    constexpr float scale = 1.0f / PCM32_MAX_ABS_VALUE;
    const float32x4_t scaleVec = vdupq_n_f32(scale);
    const float32x4_t oneVec = vdupq_n_f32(1.0f);
    const float32x4_t minusOneVec = vdupq_n_f32(-1.0f);
    size_t i = 0;
    const size_t neonSamples = numSamples - (numSamples % NEON_FLOAT32_LANES);

    for (; i < neonSamples; i += NEON_FLOAT32_LANES) {
        int32x4_t pcmVec = vld1q_s32(pcmData + i);
        float32x4_t floatVec = vcvtq_f32_s32(pcmVec);
        floatVec = vmulq_f32(floatVec, scaleVec);
        floatVec = vminq_f32(vmaxq_f32(floatVec, minusOneVec), oneVec);
        vst1q_f32(samples + i, floatVec);
    }
}
#else
void ConvertPcm32ToFloatScalar(const uint8_t* raw, float* samples, size_t numSamples)
{
    constexpr size_t bytesPerSample = 32 / 8;
    const float scale = 1.0f / PCM32_MAX_ABS_VALUE;

    for (size_t i = 0; i < numSamples; i++) {
        const size_t offset = i * bytesPerSample;

        int32_t sample = static_cast<int32_t>(raw[offset]) | (static_cast<int32_t>(raw[offset + 1]) << 8) |
                         (static_cast<int32_t>(raw[offset + 2]) << 16) | (static_cast<int32_t>(raw[offset + 3]) << 24);

        float value = static_cast<float>(sample) * scale;
        samples[i] = std::clamp(value, -1.0f, 1.0f);
    }
}
#endif

ErrorCode ConvertFloat32(const std::vector<uint8_t>& raw, std::vector<float>& samples)
{
    size_t target_size_bytes = samples.size() * sizeof(float);
    if (target_size_bytes < raw.size()) {
        return ERR_INVALID_PARAM; // Buffer too small
    }
    errno_t result = memcpy_s(samples.data(), target_size_bytes, raw.data(), target_size_bytes);
    if (result != 0) {
        LogError << "memcpy_s failed: buffer overflow or invalid parameters." << GetErrorInfo(ERR_BAD_FREE);
        return ERR_BAD_FREE;
    }
    return SUCCESS;
}

ErrorCode ConvertPcm16(const uint8_t* rawData, float* outputData, size_t expectedSize)
{
    try {
#ifdef __ARM_NEON
        ConvertPcm16ToFloatNeon(rawData, outputData, expectedSize);
#else
        ConvertPcm16ToFloatScalar(rawData, outputData, expectedSize);
#endif
    } catch (const std::exception& e) {
        return ERR_INVALID_FILE_SIZE;
    }
    return SUCCESS;
}

ErrorCode ConvertPcm24(const uint8_t* rawData, float* outputData, size_t expectedSize)
{
    try {
        ConvertPcm24ToFloatScalar(rawData, outputData, expectedSize);
    } catch (const std::exception& e) {
        return ERR_INVALID_FILE_SIZE;
    }
    return SUCCESS;
}

ErrorCode ConvertPcm32(const uint8_t* rawData, float* outputData, size_t expectedSize)
{
    try {
#ifdef __ARM_NEON
        ConvertPcm32ToFloatNeon(rawData, outputData, expectedSize);
#else
        ConvertPcm32ToFloatScalar(rawData, outputData, expectedSize);
#endif
    } catch (const std::exception& e) {
        return ERR_INVALID_FILE_SIZE;
    }
    return SUCCESS;
}

ErrorCode ReadAndCheckRiffHeader(const std::vector<uint8_t>& data, size_t& offset)
{
    constexpr size_t kRiffHeaderSize = 12;
    const size_t riffLen = 4;
    const size_t fileSizeLen = 4;
    const size_t waveLen = 4;
    const uint8_t* ptr = data.data() + offset;

    if (std::memcmp(ptr, "RIFF", riffLen) != 0 || std::memcmp(ptr + riffLen + fileSizeLen, "WAVE", waveLen) != 0) {
        LogError << "Invalid RIFF/WAVE header" << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    offset += kRiffHeaderSize;
    return SUCCESS;
}

ErrorCode FindAndReadFmtChunk(const std::vector<uint8_t>& data, size_t& offset, WavFmt& fmt)
{
    constexpr size_t kChunkIdSize = 4;
    constexpr size_t kChunkSizeField = 4;
    constexpr size_t kStandardFmtChunkSize = 16;

    while (offset + kChunkIdSize + kChunkSizeField <= data.size()) {
        const uint8_t* ptr = data.data() + offset;

        const uint8_t* chunkId = ptr;
        offset += kChunkIdSize;

        uint32_t chunkSize;
        errno_t result = memcpy_s(&chunkSize, kChunkSizeField, data.data() + offset, kChunkSizeField);
        if (result != 0) {
            LogError << "memcpy_s failed: buffer overflow or invalid parameters." << GetErrorInfo(ERR_BAD_FREE);
            return ERR_BAD_FREE;
        }
        offset += kChunkSizeField;

        if (std::memcmp(chunkId, "fmt ", kChunkIdSize) == 0) {
            if (chunkSize < kStandardFmtChunkSize || offset + kStandardFmtChunkSize > data.size()) {
                LogError << "Invalid fmt chunk size" << GetErrorInfo(ERR_INVALID_PARAM);
                return ERR_INVALID_PARAM;
            }

            errno_t result = memcpy_s(&fmt, kStandardFmtChunkSize, data.data() + offset, kStandardFmtChunkSize);
            if (result != 0) {
                LogError << "memcpy_s failed: buffer overflow or invalid parameters." << GetErrorInfo(ERR_BAD_FREE);
                return ERR_BAD_FREE;
            }
            offset += kStandardFmtChunkSize;

            if (chunkSize > kStandardFmtChunkSize) {
                offset += (chunkSize - kStandardFmtChunkSize);
            }

            return SUCCESS;
        } else {
            offset += chunkSize;
        }
    }

    LogError << "Can't find fmt chunk" << GetErrorInfo(ERR_INVALID_PARAM);
    return ERR_INVALID_PARAM;
}

ErrorCode FindDataChunk(const std::vector<uint8_t>& data, size_t& offset, uint32_t& dataSize)
{
    constexpr size_t kChunkIdSize = 4;
    constexpr size_t kChunkSizeField = 4;

    while (offset + kChunkIdSize + kChunkSizeField <= data.size()) {
        const uint8_t* ptr = data.data() + offset;

        const uint8_t* chunkId = ptr;
        offset += kChunkIdSize;

        uint32_t chunkSize;
        errno_t result = memcpy_s(&chunkSize, kChunkSizeField, data.data() + offset, kChunkSizeField);
        if (result != 0) {
            LogError << "memcpy_s failed: buffer overflow or invalid parameters." << GetErrorInfo(ERR_BAD_FREE);
            return ERR_BAD_FREE;
        }
        offset += kChunkSizeField;

        if (std::memcmp(chunkId, "data", kChunkIdSize) == 0) {
            if (offset + chunkSize > data.size()) {
                LogError << "Invalid data chunk size" << GetErrorInfo(ERR_INVALID_PARAM);
                return ERR_INVALID_PARAM;
            }

            dataSize = chunkSize;
            return SUCCESS;
        } else {
            offset += chunkSize;
        }
    }

    LogError << "Can't find data chunk" << GetErrorInfo(ERR_INVALID_PARAM);
    return ERR_INVALID_PARAM;
}

ErrorCode CheckAudioFormat(const WavFmt& fmt)
{
    if (fmt.audioFormat != WAVE_FORMAT_PCM && fmt.audioFormat != WAVE_FORMAT_IEEE_FLOAT) {
        LogError << "Unsupported compressed audio data format." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    if (fmt.bitsPerSample != BITS_PER_SAMPLE_16 && fmt.bitsPerSample != BITS_PER_SAMPLE_24 &&
        fmt.bitsPerSample != BITS_PER_SAMPLE_32) {
        LogError << "Unsupported bit depth." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

const std::vector<AudioConvertEntry> gAudioConverters = {
    {WAVE_FORMAT_PCM, 16, ConvertPcm16},
    {WAVE_FORMAT_PCM, 24, ConvertPcm24},
    {WAVE_FORMAT_PCM, 32, ConvertPcm32},
};

ErrorCode ConvertAudioDataToFloat(const std::vector<uint8_t>& rawData, const WavFmt& fmt, uint32_t expectedSize,
                                  std::vector<float>& samples)
{
    if (fmt.audioFormat == WAVE_FORMAT_IEEE_FLOAT) {
        if (fmt.bitsPerSample == BITS_PER_SAMPLE_32) {
            return ConvertFloat32(rawData, samples);
        } else {
            LogError << "IEEE_FLOAT only supports 32-bit sample data." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    } else if (fmt.audioFormat == WAVE_FORMAT_PCM) {
        for (const auto& entry : gAudioConverters) {
            if (entry.audioFormat == fmt.audioFormat && entry.bitsPerSample == fmt.bitsPerSample) {
                return entry.convert(rawData.data(), samples.data(), expectedSize);
            }
        }
    }

    LogError << "Unsupported audio format / bit depth: " << fmt.audioFormat << "/" << fmt.bitsPerSample
             << GetErrorInfo(ERR_INVALID_PARAM);
    return ERR_INVALID_PARAM;
}

} // namespace

namespace Acc {

ErrorCode CheckSingleAudioInputs(const char* path, std::optional<int> sr)
{
    if (sr.has_value()) {
        int value = sr.value();
        if (value <= 0) {
            LogError << "Sample rate must be positive." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        constexpr int kMaxSampleRate = 64000;
        if (value > kMaxSampleRate) {
            LogError << "The max sample rate supported is 64000 Hz."
                     << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    if (path == nullptr) {
        LogError << "Path is null." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    if (!CheckFileExtension(path, "wav")) {
        LogError << "Invalid audio suffix, only support 'wav', 'WAV'."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (!IsFileValid(path)) {
        LogError << "Audio path is invalid." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    std::ifstream file;
    if (!CheckFileOpen(path, file)) {
        LogError << "The file is invalid, file open failed." << GetErrorInfo(ERR_OPEN_FILE_FAILURE);
        return ERR_OPEN_FILE_FAILURE;
    }
    int64_t fileSize = 0;
    if (!CheckAndGetFileSize(file, AUDIO_MAX_FILE_SIZE, fileSize)) {
        LogError << "The file is invalid, file size out of range." << GetErrorInfo(ERR_INVALID_FILE_SIZE);
        return ERR_INVALID_FILE_SIZE;
    }

    return SUCCESS;
}

ErrorCode MixChannelsInterleaved(float* output, const float* input, size_t numFrames, int numChannels)
{
    if (numChannels == 0) {
        LogError << "Invalid audio channel number: " << numChannels << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    const float factor = 1.0f / numChannels;
    constexpr int stereoChannels = 2;

#ifdef __ARM_NEON
    if (numChannels == stereoChannels) {
        const size_t vectorizedFrames = (numFrames / stereoChannels) * stereoChannels;

        for (size_t i = 0; i < vectorizedFrames; i += stereoChannels) {
            float32x4_t quad = vld1q_f32(input + i * stereoChannels);
            float32x2_t low = vget_low_f32(quad);   // [L1, R1]
            float32x2_t high = vget_high_f32(quad); // [L2, R2]
            // Deinterleave
            float32x2x2_t deinterleaved = vuzp_f32(low, high);
            float32x2_t left = deinterleaved.val[0];
            float32x2_t right = deinterleaved.val[1];
            float32x2_t result = vmul_f32(vadd_f32(left, right), vdup_n_f32(factor));

            vst1_f32(output + i, result);
        }

        // tail
        for (size_t i = vectorizedFrames; i < numFrames; i++) {
            output[i] = (input[i * stereoChannels] + input[i * stereoChannels + 1]) * factor;
        }
    }
#else
    // Generic multi-channel mixing
    for (size_t i = 0; i < numFrames; i++) {
        float sum = 0.0f;
        for (int c = 0; c < numChannels; c++) {
            sum += input[i * numChannels + c];
        }
        output[i] = sum * factor;
    }
#endif
    return SUCCESS;
}

ErrorCode AudioDecode(const char* filePath, AudioData& outputAudioData)
{
    std::vector<uint8_t> fileData;

    ErrorCode ret = ReadFile(filePath, fileData, AUDIO_MAX_FILE_SIZE);
    if (ret != SUCCESS) {
        return ret;
    }

    size_t offset = 0;
    ret = ReadAndCheckRiffHeader(fileData, offset);
    if (ret != SUCCESS)
        return ret;

    WavFmt fmt;
    ret = FindAndReadFmtChunk(fileData, offset, fmt);
    if (ret != SUCCESS)
        return ret;

    ret = CheckAudioFormat(fmt);
    if (ret != SUCCESS)
        return ret;

    uint32_t dataSize;
    ret = FindDataChunk(fileData, offset, dataSize);
    if (ret != SUCCESS)
        return ret;

    std::vector<uint8_t> rawData(fileData.begin() + offset, fileData.begin() + offset + dataSize);
    const uint32_t bytesPerSample = fmt.bitsPerSample / 8;
    const uint32_t numSamples = dataSize / (bytesPerSample * fmt.numChannels);

    std::vector<float> samples(numSamples * fmt.numChannels);
    ret = ConvertAudioDataToFloat(rawData, fmt, numSamples, samples);
    if (ret != SUCCESS)
        return ret;

    outputAudioData = {std::move(samples), fmt.sampleRate, fmt.numChannels, fmt.bitsPerSample, numSamples};

    return SUCCESS;
}
} // namespace Acc