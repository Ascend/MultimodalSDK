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
 * Description: Processing of the video.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/video/Video.h"

#include <chrono>
#include <array>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <map>
#include <variant>
#include <cmath>
#include <future>
#include <condition_variable>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "acc/utils/LogImpl.h"
#include "acc/ErrorCode.h"
#include "acc/utils/FileUtils.h"
#include "acc/utils/ThreadPool.h"
#include "acc/utils/VideoUtils.h"
#include "acc/ErrorCode.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace {
using namespace Acc;

// decode constraints
constexpr uint32_t MAX_VIDEO_DECODE_CHANNELS = 4; // Max video decode chn num
constexpr uint32_t DISPLAY_FRAME_NUM = 2;         // Number of display frames [0, 16]
constexpr uint32_t REF_FRAME_NUM = 8;             // Number of reference frames [0, 16]
constexpr uint32_t OUTPUT_RGB_CHANNELS = 3;       // Output frame chn num
constexpr uint32_t OUTPUT_BUFFER_NUM = 8;         // Number of output buffer for each chn
constexpr uint32_t OUTPUT_WIDTH = 0;              // Output image width, support resize, 0 for not resize
constexpr uint32_t OUTPUT_HEIGHT = 0;             // Output image height, support resize, 0 for not resize
constexpr uint32_t TIMEOUT = 1000;                // Default timeout
constexpr uint32_t OUT_FRAME_FLAG_0 = 0;          // Decode success
constexpr uint32_t OUT_FRAME_FLAG_1 = 1;          // Decode failed
constexpr uint32_t OUT_FRAME_FLAG_2 = 2;          // Decode success but no picture
constexpr uint32_t OUT_FRAME_FLAG_3 = 3;          // Decode failed, ref frame num error
constexpr uint32_t OUT_FRAME_FLAG_4 = 4;          // Decode failed, frame buf size error
constexpr float_t BUF_SIZE_WEIGHT = 1.5;          // Weight for buf size, 1.5 means 3 / 2
constexpr auto GET_AVAILABLE_CHN_TIMEOUT = std::chrono::seconds(15); // Max waiting time for get available chnId

/**
 * @description: Checks whether the frame idx set is valid and fills it when it's empty.
 * @param nFrames: Total frames count
 * @param frameIndices: Set of frame index
 * @return int: Error code
 */
ErrorCode CheckTargetFrameIndices(uint32_t nFrames, std::set<uint32_t>& frameIndices, int sampleNum)
{
    if (nFrames == 0) {
        LogError << "The frame number for decoding video must be greater than zero." << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    if (frameIndices.empty() && sampleNum == -1) {
        LogError << "the target frame indices or target frame number must be set." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (!frameIndices.empty()) {
        uint32_t minFrameId = *frameIndices.begin();
        uint32_t maxFrameId = *frameIndices.rbegin();
        if (maxFrameId >= nFrames) {
            LogError << "Target frame idx must be in the range of [0, " << nFrames << "), but get idx in the range of ["
                     << minFrameId << ", " << maxFrameId << "]" << GetErrorInfo(ERR_OUT_OF_RANGE);
            return ERR_OUT_OF_RANGE;
        }
        return SUCCESS;
    }
    if (static_cast<uint32_t>(sampleNum) > nFrames || sampleNum < 1) {
        LogError << "the target frame number must be less than " << nFrames << " and greater than 1"
                 << GetErrorInfo(ERR_OUT_OF_RANGE);
        return ERR_OUT_OF_RANGE;
    }
    if (sampleNum == 1) {
        frameIndices.insert(0);
        return SUCCESS;
    }
    uint32_t validSampleNum = static_cast<uint32_t>(sampleNum);
    for (uint32_t i = 0; i < validSampleNum; i++) {
        uint32_t idx = static_cast<uint32_t>(i * (nFrames - 1) / (validSampleNum - 1));
        frameIndices.insert(idx);
    }
    return SUCCESS;
}

ErrorCode CheckVideoResolution(AVStream* videoStream)
{
    if (videoStream == nullptr) {
        LogError << "CheckVideo resolution failed, AVStream is nullptr." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    uint32_t width = static_cast<uint32_t>(videoStream->codecpar->width);
    uint32_t height = static_cast<uint32_t>(videoStream->codecpar->height);
    if (width < MIN_STREAM_WIDTH || width > MAX_STREAM_WIDTH || height < MIN_STREAM_HEIGHT ||
        height > MAX_STREAM_HEIGHT) {
        LogError << "Input Video resolution is invalid. Width and height must be in the range of [" << MIN_STREAM_WIDTH
                 << ", " << MIN_STREAM_HEIGHT << "] to [" << MAX_STREAM_WIDTH << ", " << MAX_STREAM_HEIGHT
                 << "], but get [" << width << ", " << height << "]" << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode InitVideoInfo(const char* path, std::set<uint32_t>& targetIndices, int sampleNum, VideoAuxInfo& videoAuxInfo,
                        std::vector<int>& targetKeyframeIndices)
{
    AVFormatContext* formatCtx = nullptr;

    // open video file
    if (avformat_open_input(&formatCtx, path, nullptr, nullptr) != SUCCESS) {
        LogError << "Cannot open video file, please ensure that the input video is legal."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        avformat_close_input(&formatCtx);
        LogError << "Cannot find video stream information, please ensure that the input video is legal."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }

    int videoStreamIdx = FindVideoStream(formatCtx);
    if (videoStreamIdx == -1) {
        avformat_close_input(&formatCtx);
        LogError << "No video stream found, please ensure that the input video is legal."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }

    AVStream* videoStream = formatCtx->streams[videoStreamIdx];
    ErrorCode ret = CheckVideoResolution(videoStream);
    if (ret != SUCCESS) {
        avformat_close_input(&formatCtx);
        return ret;
    }

    double originFps = 0;
    int64_t totalFrames = 0;
    ret = GetFramesAndFPS(videoStream, originFps, totalFrames);
    if (ret != SUCCESS) {
        avformat_close_input(&formatCtx);
        LogError << "Cannot determine video frame count or FPS, may caused by input video which is broken."
                 << GetErrorInfo(ret);
        return ret;
    }
    ret = CheckTargetFrameIndices(totalFrames, targetIndices, sampleNum);
    if (ret != SUCCESS) {
        avformat_close_input(&formatCtx);
        return ret;
    }
    ret = ConstructVideoAuxInfo(formatCtx, videoStreamIdx, targetIndices, videoAuxInfo);
    if (ret != SUCCESS) {
        avformat_close_input(&formatCtx);
        return ret;
    }
    // close formatCtx, open in the parallel decode threads later
    avformat_close_input(&formatCtx);
    // get keyframes indices
    targetKeyframeIndices = videoAuxInfo.GetTargetKeyframeIndices();

    return SUCCESS;
}

ErrorCode DecodeKeyframesParallel(const char* path, VideoAuxInfo& videoAuxInfo,
                                  const std::vector<int>& targetKeyframeIndices, std::map<int, AVFrame*>& results)
{
    ThreadPool& pool = ThreadPool::GetInstance();
    std::atomic<bool> errorOccurred(false);
    std::mutex resultsMutex;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < targetKeyframeIndices.size(); i++) {
        // any errors, terminate the task early
        if (errorOccurred.load()) {
            break;
        }
        futures.push_back(
            pool.Submit([i, &targetKeyframeIndices, &videoAuxInfo, path, &results, &errorOccurred, &resultsMutex]() {
                std::map<int, AVFrame*> threadResult;
                auto ret = VideoDecodeSeek(path, targetKeyframeIndices[i], videoAuxInfo, threadResult);
                if (ret != SUCCESS) {
                    errorOccurred.store(true);
                    throw std::runtime_error("VideoDecodeSeek failed with code: " + std::to_string(ret));
                }
                std::lock_guard<std::mutex> lock(resultsMutex);
                for (auto& [index, frame] : threadResult) {
                    results[index] = frame;
                }
            }));
    }
    ErrorCode finalStatus = SUCCESS;
    for (auto& future : futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            LogError << "Thread terminated with exception." << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
            finalStatus = ERR_FFMPEG_COMMON_FAILURE;
        }
    }
    if (finalStatus != SUCCESS) {
        for (auto& [index, frame] : results) {
            av_frame_free(&frame);
        }
        results.clear();
    }
    return finalStatus;
}

ErrorCode ConvertFramesToImage(const std::map<int, AVFrame*>& results, const std::set<uint32_t>& targetIndices,
                               std::vector<Image>& frames)
{
    std::vector<AVFrame*> rgbFrames;
    rgbFrames.reserve(targetIndices.size());
    for (int index : targetIndices) {
        auto it = results.find(index);
        rgbFrames.push_back(it != results.end() ? it->second : nullptr);
    }

    frames.reserve(rgbFrames.size());
    const char* device = "cpu";
    ErrorCode ret = SUCCESS;
    for (AVFrame* rgbFrame : rgbFrames) {
        if (!rgbFrame || rgbFrame->format != AV_PIX_FMT_RGB24) {
            LogError << "The decoded frame is nullptr or not rgb24, may caused by raw input video"
                     << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
            av_frame_free(&rgbFrame);
            ret = ERR_FFMPEG_COMMON_FAILURE;
            continue;
        }
        std::vector<size_t> imSize = {static_cast<size_t>(rgbFrame->width), static_cast<size_t>(rgbFrame->height)};
        std::shared_ptr<uint8_t> framePtr(rgbFrame->data[0], [](uint8_t* ptr) {
            if (ptr) {
                av_free(ptr);
            }
        });
        Image img(framePtr, imSize, ImageFormat::RGB, DataType::UINT8, device);
        rgbFrame->buf[0] = nullptr;
        frames.push_back(img);
        av_frame_free(&rgbFrame);
    }
    if (ret != SUCCESS) {
        frames.clear();
    }
    return ret;
}

} // namespace

namespace Acc {
/**
 * @description: Video decode by cpu
 * @param path: Input video path for mp4
 * @param frames: Decoded output frames
 * @param targetIndices: Target frame indices
 * @return int: Error code
 */
ErrorCode VideoDecodeCpu(const char* path, std::vector<Image>& frames, const std::set<uint32_t>& targetIndices,
                         int sampleNum)
{
    frames.clear();
    VideoAuxInfo videoAuxInfo;
    std::vector<int> targetKeyframeIndices;

    std::set<uint32_t> checkedTargetIndices = targetIndices;
    auto ret = InitVideoInfo(path, checkedTargetIndices, sampleNum, videoAuxInfo, targetKeyframeIndices);
    if (ret != SUCCESS) {
        return ret;
    }

    std::map<int, AVFrame*> results;
    ret = DecodeKeyframesParallel(path, videoAuxInfo, targetKeyframeIndices, results);
    if (ret != SUCCESS) {
        return ret;
    }

    return ConvertFramesToImage(results, checkedTargetIndices, frames);
}

ErrorCode VideoDecode(const char* path, const char* device, std::vector<Image>& frames,
                      const std::set<uint32_t>& frameIndices, int sampleNum)
{
    if (!device || strcmp(device, "cpu") != 0) {
        LogError << "Illegal device. Only 'cpu' is supported now." << GetErrorInfo(ERR_UNSUPPORTED_TYPE);
        return ERR_UNSUPPORTED_TYPE;
    }
    // check video path
    if (!IsFileValid(path)) {
        LogError << "Video decode failed, video path is invalid." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }

    ErrorCode ret = SUCCESS;
    // cpu video decode
    if (!CheckFileExtension(path, "mp4")) {
        LogError << "Video decode failed, invalid video suffix, only support 'mp4', 'MP4'."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    ret = VideoDecodeCpu(path, frames, frameIndices, sampleNum);
    if (ret != SUCCESS) {
        LogError << "Video decode failed.";
    }
    return ret;
}
} // namespace Acc
