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
 * Description: audio api test file.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#include <vector>
#include <gtest/gtest.h>
#include <dirent.h>
#include <sys/stat.h>
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/audio/Audio.h"

using namespace Acc;
using namespace std;

namespace {
constexpr int SAMPLE_RATE = 16000;

class AudioTest : public testing::Test {
public:
    // Test audio file paths
    const char* validAudioPath_ = "../../data/audios/audio_test.wav";
    const char* invalidSuffixAudioPath_ = "../../data/audios/audio_test.mp3";
    const char* nonexistentAudioPath_ = "../../data/audios/nonexistent.wav";
    const char* emptyAudioPath_ = "../../data/audios/audio_empty.wav";
    const char* pcm24MonoPath_     = "../../data/audios/test_pcm24_mono.wav";
    const char* pcm32MonoPath_     = "../../data/audios/test_pcm32_mono.wav";
    const char* float32MonoPath_   = "../../data/audios/test_float32_mono.wav";
    const char* pcm16StereoPath_   = "../../data/audios/test_pcm16_stereo.wav";

protected:
    void SetUp() override
    {
        chmod(validAudioPath_, 0440);
        chmod(invalidSuffixAudioPath_, 0440);
        chmod(emptyAudioPath_, 0440);
        chmod(pcm24MonoPath_, 0440);
        chmod(pcm32MonoPath_, 0440);
        chmod(float32MonoPath_, 0440);
        chmod(pcm16StereoPath_, 0440);
    }
};

TEST_F(AudioTest, LoadAudioSingle_ShouldSucceed_WhenAudioIsValid)
{
    Tensor tensor;
    int original_sr;

    ErrorCode result = LoadAudioSingle(validAudioPath_, tensor, original_sr, SAMPLE_RATE);

    EXPECT_EQ(result, SUCCESS);
    EXPECT_GT(tensor.NumBytes(), 0);
}

TEST_F(AudioTest, LoadAudioBatch_ShouldSucceed_WhenAudiosAreValid)
{
    std::vector<std::string> audioPaths = {validAudioPath_, validAudioPath_};
    std::vector<Tensor> tensors(audioPaths.size());
    std::vector<int> originalSrs;

    ErrorCode ret = LoadAudioBatch(audioPaths, tensors, originalSrs, SAMPLE_RATE);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(tensors.size(), audioPaths.size());
    EXPECT_EQ(originalSrs.size(), audioPaths.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        EXPECT_GT(tensors[i].NumBytes(), 0);
    }
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenAudioPathIsInvalid)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result = LoadAudioSingle(nonexistentAudioPath_, tensor, oringal_sr, SAMPLE_RATE);

    EXPECT_NE(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenSampleRateIsInvalid)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result1 = LoadAudioSingle(validAudioPath_, tensor, oringal_sr, 0);
    EXPECT_NE(result1, SUCCESS);

    ErrorCode result2 = LoadAudioSingle(validAudioPath_, tensor, oringal_sr, -44100);
    EXPECT_NE(result2, SUCCESS);

    ErrorCode result3 = LoadAudioSingle(validAudioPath_, tensor, oringal_sr, 65000);
    EXPECT_NE(result3, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenAudioDataIsInvalid)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result = LoadAudioSingle(emptyAudioPath_, tensor, oringal_sr, SAMPLE_RATE);
    EXPECT_NE(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenFileFormatIsUnsupported)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result = LoadAudioSingle(invalidSuffixAudioPath_, tensor, oringal_sr, SAMPLE_RATE);
    EXPECT_NE(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldSucceed_ForPcm24Mono)
{
    Tensor tensor;
    int oringal_sr;
    ErrorCode result = LoadAudioSingle(pcm24MonoPath_, tensor, oringal_sr, std::nullopt);
    EXPECT_EQ(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldSucceed_ForPcm32Mono)
{
    Tensor tensor;
    int oringal_sr;
    ErrorCode result = LoadAudioSingle(pcm32MonoPath_, tensor, oringal_sr, std::nullopt);
    EXPECT_EQ(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldSucceed_ForFloat32Mono)
{
    Tensor tensor;
    int oringal_sr;
    ErrorCode result = LoadAudioSingle(float32MonoPath_, tensor, oringal_sr, std::nullopt);
    EXPECT_EQ(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldSucceed_ForStereoAndMixToMono)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result = LoadAudioSingle(pcm16StereoPath_, tensor, oringal_sr, std::nullopt);
    EXPECT_EQ(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioBatch_ShouldReturnError_WhenBatchSizeExceedsLimit)
{
    std::vector<std::string> audioPaths(129, validAudioPath_);
    std::vector<Tensor> tensors;
    std::vector<int> originalSrs;

    ErrorCode ret = LoadAudioBatch(audioPaths, tensors, originalSrs, SAMPLE_RATE);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(AudioTest, LoadAudioBatch_ShouldReturnError_WhenAnyAudioPathIsInvalid)
{
    std::vector<std::string> audioPaths = {validAudioPath_, nonexistentAudioPath_, validAudioPath_};
    std::vector<Tensor> tensors;
    std::vector<int> originalSrs;

    ErrorCode ret = LoadAudioBatch(audioPaths, tensors, originalSrs, SAMPLE_RATE);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}