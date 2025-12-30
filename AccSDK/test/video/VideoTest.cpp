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
 * Description: video api test file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include <set>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/video/Video.h"

using namespace Acc;
using namespace std;
namespace {
constexpr int DEFAULT_SAMPLE_COUNT = 32;
constexpr uint32_t NUM_DEVICES = 4;
constexpr uint32_t NUM_THREADS = 4;
constexpr size_t EXPECTED_WIDTH = 1920;
constexpr size_t EXPECTED_HEIGHT = 1080;
constexpr size_t EXPECTED_FRAMES = 48;
const char* VIDEO_PATH_WITHOUT_VIDEO_STREAM = "../../data/videos/test_acc.mp4";
const char* DEVICE_CPU = "cpu";
const char* DEVICE_NPU = "npu";

class VideoTest : public testing::Test {
public:
    std::set<uint32_t> FrameSampleEqualInterval(int startIdx, int endIdx, int sampleCount = DEFAULT_SAMPLE_COUNT)
    {
        std::set<uint32_t> indices;
        int totalFrames = endIdx - startIdx + 1;
        int step = totalFrames / (sampleCount - 1);
        for (int i = 0; i < sampleCount; i++) {
            int currentIndex = startIdx + i * step;
            if (currentIndex > endIdx) {
                currentIndex = endIdx;
            }
            indices.insert(currentIndex);
        }
        return indices;
    }
    std::string validVideoPath_ = "../../data/videos/video_1min_30fps.mp4";
    std::string invalidSuffixVideoPath_ = "../../data/videos/video_200weight.avi";
    std::string invalidSizeVideoPath_ = "../../data/videos/video_200weight.mp4";

protected:
    void SetUp() override
    {
        chmod(VIDEO_PATH_WITHOUT_VIDEO_STREAM, 0440);
    }
};

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenFilePermissionsInvalid)
{
    std::vector<Image> frames;
    // construct indices, take 32 frames at equal intervals
    int startIdx = 0;
    int endIdx = 250;
    std::set<uint32_t> indices = FrameSampleEqualInterval(startIdx, endIdx);
    chmod(validVideoPath_.c_str(), 0777);
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, indices);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnSuccess_WhenDeviceIsCpuAndDecodeSucceeds)
{
    std::vector<Image> frames;
    // construct indices, take 32 frames at equal intervals
    int startIdx = 0;
    int endIdx = 250;
    std::set<uint32_t> indices = FrameSampleEqualInterval(startIdx, endIdx);
    chmod(validVideoPath_.c_str(), 0440);
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, indices);

    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(frames.size(), DEFAULT_SAMPLE_COUNT);

    for (const auto& frame : frames) {
        EXPECT_EQ(frame.Width(), EXPECTED_WIDTH);
        EXPECT_EQ(frame.Height(), EXPECTED_HEIGHT);
        EXPECT_EQ(frame.Format(), ImageFormat::RGB);
    }
}

TEST_F(VideoTest, VideoDecode_ShouldReturnInvalidParam_WhenPathIsInvalid)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2};
    const char* invalidVideoPath_ = "invalid_path.mp4";
    ErrorCode ret = VideoDecode(invalidVideoPath_, DEVICE_CPU, frames, targetIndices);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFail_WhenFileDoesNotContainVideoStream)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2};
    ErrorCode ret = VideoDecode(VIDEO_PATH_WITHOUT_VIDEO_STREAM, DEVICE_CPU, frames, targetIndices);
    EXPECT_EQ(ret, ERR_FFMPEG_COMMON_FAILURE);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnInvalidParam_WhenSampleNumBiggerThanTotalFrames)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 2000);
    EXPECT_EQ(ret, ERR_OUT_OF_RANGE);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFramesCountsEqualWithTargetIndices_WhenParamIsValid)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 33};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(frames.size(), targetIndices.size());
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFramesCountsEqualWithTargetIndices_WhenBothSampleNumIsProvided)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 33};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 1200);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(frames.size(), targetIndices.size());
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnInvalidParam_WhenBothSampleNumAndTargetIndicesEmpty)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnSuccess_WhenSampleNumEqualsOne)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 1);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(frames.size(), 1);
    for (const auto& frame : frames) {
        EXPECT_EQ(frame.Width(), EXPECTED_WIDTH);
        EXPECT_EQ(frame.Height(), EXPECTED_HEIGHT);
        EXPECT_EQ(frame.Format(), ImageFormat::RGB);
    }
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnSuccess_WhenSampleNumIsValidAndTargetIndicesEmpty)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(frames.size(), 20);
    for (const auto& frame : frames) {
        EXPECT_EQ(frame.Width(), EXPECTED_WIDTH);
        EXPECT_EQ(frame.Height(), EXPECTED_HEIGHT);
        EXPECT_EQ(frame.Format(), ImageFormat::RGB);
    }
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenTargetIndicesOutOfRange)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 2000};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_OUT_OF_RANGE);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenDeviceIsNpu)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 200};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_NPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_UNSUPPORTED_TYPE);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenDeviceIsNullptr)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 200};
    int ret = VideoDecode(validVideoPath_.c_str(), nullptr, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_UNSUPPORTED_TYPE);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenTargetIndicesOutOfRangeUsingBorderValue)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 1802};
    ErrorCode ret = VideoDecode(validVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_OUT_OF_RANGE);
}

TEST_F(VideoTest, VideoDecode_ShouldReturnFailed_WhenInvalidExtension)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10};
    ErrorCode ret = VideoDecode(invalidSuffixVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenVideoSizeOutOfRange)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10};
    ErrorCode ret = VideoDecode(invalidSizeVideoPath_.c_str(), DEVICE_CPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}


TEST_F(VideoTest, VideoDecodeOnCpu_ShouldReturnFailed_WhenPathIsNullptr)
{
    std::vector<Image> frames;
    std::set<uint32_t> targetIndices = {0, 1, 2, 10, 22, 200};
    int ret = VideoDecode(nullptr, DEVICE_NPU, frames, targetIndices, 20);
    EXPECT_EQ(ret, ERR_UNSUPPORTED_TYPE);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    ErrorCode ret = RUN_ALL_TESTS();
    return ret;
}