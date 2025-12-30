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
 * Description: video utils api test file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include <gtest/gtest.h>
#include "acc/ErrorCode.h"
#define private public
#include "acc/utils/VideoUtils.h"
#undef private

using namespace Acc;
using namespace std;

namespace {
constexpr int INVALID_VIDEO_STREAM_INDEX = -1;
constexpr uint32_t INDEX_0 = 0;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_4 = 4;
constexpr double FRAME_RATE = 30.0;
constexpr double TIME_BASE = 1.0;
constexpr int TOTAL_FRAMES_NUM = 100;
constexpr int PROPORTIONAL_FACTOR = 100;
constexpr int DURATION_TIME = 100;
class VideoUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        formatCtx = new AVFormatContext();
        formatCtx->nb_streams = 0;
        formatCtx->streams = nullptr;
        videoStream = new AVStream();
    }

    void TearDown() override
    {
        if (formatCtx) {
            for (int i = 0; i < formatCtx->nb_streams; i++) {
                delete formatCtx->streams[i];
            }
            delete[] formatCtx->streams;
            delete formatCtx;
        }
        delete videoStream;
    }

    AVFormatContext* formatCtx;
    AVStream* videoStream;
    std::string validVideoPath = "../../data/videos/video_1min_30fps.mp4";
};

TEST_F(VideoUtilsTest, FindVideoStream_ShouldReturnInvalidIndex_WhenFormatCtxIsNull)
{
    AVFormatContext* nullFormatCtx = nullptr;
    EXPECT_EQ(FindVideoStream(nullFormatCtx), INVALID_VIDEO_STREAM_INDEX);
}

TEST_F(VideoUtilsTest, FindVideoStream_ShouldReturnInvalidIndex_WhenNoVideoStream)
{
    int streamCount = 2;
    formatCtx->nb_streams = streamCount;
    formatCtx->streams = new AVStream *[streamCount];
    for (int i = 0; i < streamCount; i++) {
        formatCtx->streams[i] = new AVStream();
        formatCtx->streams[i]->codecpar = new AVCodecParameters();
        formatCtx->streams[i]->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    }
    EXPECT_EQ(FindVideoStream(formatCtx), INVALID_VIDEO_STREAM_INDEX);
}

TEST_F(VideoUtilsTest, FindVideoStream_ShouldReturnVideoStreamIndex_WhenVideoStreamExists)
{
    int streamCount = 3;
    formatCtx->nb_streams = streamCount;
    formatCtx->streams = new AVStream *[streamCount];
    int videoStreamIdx = 1;
    for (int i = 0; i < streamCount; i++) {
        formatCtx->streams[i] = new AVStream();
        formatCtx->streams[i]->codecpar = new AVCodecParameters();
        if (i == videoStreamIdx) {
            formatCtx->streams[i]->codecpar->codec_type = AVMEDIA_TYPE_VIDEO; // Video stream
        } else {
            formatCtx->streams[i]->codecpar->codec_type = AVMEDIA_TYPE_AUDIO; // Non-video stream
        }
    }
    EXPECT_EQ(FindVideoStream(formatCtx), videoStreamIdx);
}

TEST_F(VideoUtilsTest, GetFramesAndFPS_ShouldReturnInvalidParam_WhenVideoStreamIsNull)
{
    double originFps = 0.0;
    int64_t totalFrames = 0;

    EXPECT_EQ(GetFramesAndFPS(nullptr, originFps, totalFrames), ERR_INVALID_POINTER);
}

TEST_F(VideoUtilsTest, GetFramesAndFPS_ShouldReturnSuccess_WhenFramesAndFpsAreValid)
{
    videoStream->nb_frames = TOTAL_FRAMES_NUM;
    videoStream->avg_frame_rate = av_d2q(FRAME_RATE, PROPORTIONAL_FACTOR);

    double originFps = 0.0;
    double expectedFps = 30.0;
    int64_t totalFrames = 0;
    int64_t expectedTotalFrames = 100;

    EXPECT_EQ(GetFramesAndFPS(videoStream, originFps, totalFrames), SUCCESS);
    EXPECT_EQ(totalFrames, expectedTotalFrames);
    EXPECT_EQ(originFps, expectedFps);
}

TEST_F(VideoUtilsTest, GetFramesAndFPS_ShouldRecalculateFps_WhenAvgFpsIsInvalid)
{
    videoStream->nb_frames = TOTAL_FRAMES_NUM;
    videoStream->avg_frame_rate = av_d2q(0.0, PROPORTIONAL_FACTOR);

    videoStream->r_frame_rate = av_d2q(0.0, PROPORTIONAL_FACTOR);
    videoStream->duration = DURATION_TIME;
    videoStream->time_base = av_d2q(TIME_BASE, PROPORTIONAL_FACTOR);

    double originFps = 0.0;
    double expectedFps = 1.0;
    int64_t totalFrames = 0;

    EXPECT_EQ(GetFramesAndFPS(videoStream, originFps, totalFrames), SUCCESS);
    EXPECT_EQ(totalFrames, TOTAL_FRAMES_NUM);
    EXPECT_EQ(originFps, expectedFps);
}

TEST_F(VideoUtilsTest, GetFramesAndFPS_ShouldRecalculateFrames_WhenFramesAreInvalid)
{
    videoStream->nb_frames = 0;
    videoStream->avg_frame_rate = av_d2q(FRAME_RATE, PROPORTIONAL_FACTOR);
    videoStream->duration = DURATION_TIME;
    videoStream->time_base = av_d2q(TIME_BASE, PROPORTIONAL_FACTOR);

    double originFps = 0.0;
    double expectedFps = 30.0;
    int64_t totalFrames = 0;
    int64_t expectedTotalFrames = 3000;

    EXPECT_EQ(GetFramesAndFPS(videoStream, originFps, totalFrames), SUCCESS);
    EXPECT_EQ(totalFrames, expectedTotalFrames);
    EXPECT_EQ(originFps, expectedFps);
}

TEST_F(VideoUtilsTest, GetFramesAndFPS_ShouldReturnCommFailure_WhenAllConditionsFail)
{
    videoStream->nb_frames = 0;
    videoStream->avg_frame_rate = av_d2q(0.0, PROPORTIONAL_FACTOR);
    videoStream->r_frame_rate = av_d2q(0.0, PROPORTIONAL_FACTOR);
    videoStream->duration = AV_NOPTS_VALUE;

    double originFps = 0.0;
    int64_t totalFrames = 0;

    EXPECT_EQ(GetFramesAndFPS(videoStream, originFps, totalFrames), ERR_FFMPEG_COMMON_FAILURE);
}

TEST_F(VideoUtilsTest, ConstructVideoAuxInfo_ShouldReturnCorrectVideoAuxInfo_WhenInputIsValid)
{
    AVFormatContext* formatCtx = nullptr;
    avformat_open_input(&formatCtx, validVideoPath.c_str(), nullptr, nullptr);
    int videoStreamIdx = 0;
    std::set<uint32_t> targetIndices = {INDEX_0, INDEX_2, INDEX_4};
    // Call the function under test
    VideoAuxInfo result;
    ErrorCode ret = ConstructVideoAuxInfo(formatCtx, videoStreamIdx, targetIndices, result);
    EXPECT_EQ(ret, 0);
    // Verify the result
    EXPECT_FALSE(result.keyframesAllPts.empty());
    EXPECT_FALSE(result.ptsToFrameIdx.empty());
    EXPECT_EQ(result.targetIndices, targetIndices);
    EXPECT_FALSE(result.keyframeIdxToPts.empty());
    EXPECT_FALSE(result.targetIdxToKeyframeIdx.empty());
    EXPECT_FALSE(result.keyframesTargetIndices.empty());

    // free context
    avformat_free_context(formatCtx);
}

TEST_F(VideoUtilsTest, VideoDecodeSeek_ShouldReturnFailure_WhenFileOpenFails)
{
    std::string invalidFile = "invalid_file.mp4";
    int keyframeIdx = 0;
    VideoAuxInfo videoAuxInfo;
    std::map<int, AVFrame*> results;

    int ret = VideoDecodeSeek(invalidFile, keyframeIdx, videoAuxInfo, results);
    EXPECT_EQ(ret, ERR_FFMPEG_COMMON_FAILURE);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}