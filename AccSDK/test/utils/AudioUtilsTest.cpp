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
 * Description: audio utils api test file.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <optional>
#include "acc/ErrorCode.h"
#include "acc/utils/AudioUtils.h"

using namespace Acc;
using namespace std;

namespace {

class AudioUtilsTest : public testing::Test {
public:
    const char* _validAudioPath = "../../data/audios/audio_test.wav";
    const char* _invalidSuffixAudioPath = "../../data/audios/audio_test.mp3";
    const char* _nonexistentAudioPath = "../../data/audios/nonexistent.wav";
    const char* _emptyAudioPath = "../../data/audios/audio_empty.wav";
    const int _sampleRate = 16000;

protected:
    void SetUp() override
    {
        chmod(_validAudioPath, 0440);
        chmod(_invalidSuffixAudioPath, 0440);
        chmod(_emptyAudioPath, 0440);
    }
};

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnSuccess_WhenAllChecksPass)
{
    ErrorCode ret = CheckSingleAudioInputs(_validAudioPath, _sampleRate);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnErr_WhenSampleRateIsNonPositive)
{
    int sr = 0;
    EXPECT_EQ(CheckSingleAudioInputs(_validAudioPath, sr), ERR_INVALID_PARAM);

    sr = -16000;
    EXPECT_EQ(CheckSingleAudioInputs(_validAudioPath, sr), ERR_INVALID_PARAM);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnErr_WhenSampleRateExceedsMax)
{
    int sr = 64001;
    EXPECT_EQ(CheckSingleAudioInputs(_validAudioPath, sr), ERR_INVALID_PARAM);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnErr_WhenPathIsNull)
{
    EXPECT_EQ(CheckSingleAudioInputs(nullptr, _sampleRate), ERR_INVALID_POINTER);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnErr_WhenFileExtensionIsNotWav)
{
    EXPECT_EQ(CheckSingleAudioInputs(_invalidSuffixAudioPath, _sampleRate), ERR_INVALID_PARAM);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnErr_WhenFileDoesNotExist)
{
    EXPECT_EQ(CheckSingleAudioInputs(_nonexistentAudioPath, _sampleRate), ERR_INVALID_PARAM);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldReturnErr_WhenFileIsEmpty)
{
    EXPECT_EQ(CheckSingleAudioInputs(_emptyAudioPath, _sampleRate), ERR_INVALID_FILE_SIZE);
}

TEST_F(AudioUtilsTest, CheckSingleAudioInputs_ShouldSucceed_WhenSrIsNullopt)
{
    EXPECT_EQ(CheckSingleAudioInputs(_validAudioPath), SUCCESS);
}

} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}