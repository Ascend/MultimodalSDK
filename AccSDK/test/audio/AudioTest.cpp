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
    const char* _validAudioPath = "../../data/audios/audio_test.wav";
    const char* _invalidSuffixAudioPath = "../../data/audios/audio_test.mp3";
    const char* _nonexistentAudioPath = "../../data/audios/nonexistent.wav";
    const char* _emptyAudioPath = "../../data/audios/audio_empty.wav";

protected:
    void SetUp() override
    {
        chmod(_validAudioPath, 0440);
        chmod(_invalidSuffixAudioPath, 0440);
        chmod(_emptyAudioPath, 0440);
    }
};

TEST_F(AudioTest, LoadAudioSingle_ShouldSucceed_WhenAudioIsValid)
{
    Tensor tensor;
    int original_sr;

    ErrorCode result = LoadAudioSingle(_validAudioPath, tensor, original_sr, SAMPLE_RATE);

    EXPECT_EQ(result, SUCCESS);
    EXPECT_GT(tensor.NumBytes(), 0);
}

TEST_F(AudioTest, LoadAudioBatch_ShouldSucceed_WhenAudiosAreValid)
{
    std::vector<std::string> audioPaths = {_validAudioPath, _validAudioPath};
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

    ErrorCode result = LoadAudioSingle(_nonexistentAudioPath, tensor, oringal_sr, SAMPLE_RATE);

    EXPECT_NE(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenSampleRateIsInvalid)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result1 = LoadAudioSingle(_validAudioPath, tensor, oringal_sr, 0);
    EXPECT_NE(result1, SUCCESS);

    ErrorCode result2 = LoadAudioSingle(_validAudioPath, tensor, oringal_sr, -44100);
    EXPECT_NE(result2, SUCCESS);

    ErrorCode result3 = LoadAudioSingle(_validAudioPath, tensor, oringal_sr, 65000);
    EXPECT_NE(result3, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenAudioDataIsInvalid)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result = LoadAudioSingle(_emptyAudioPath, tensor, oringal_sr, SAMPLE_RATE);
    EXPECT_NE(result, SUCCESS);
}

TEST_F(AudioTest, LoadAudioSingle_ShouldReturnError_WhenFileFormatIsUnsupported)
{
    Tensor tensor;
    int oringal_sr;

    ErrorCode result = LoadAudioSingle(_invalidSuffixAudioPath, tensor, oringal_sr, SAMPLE_RATE);
    EXPECT_NE(result, SUCCESS);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}