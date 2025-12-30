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
 * Description: Video utils file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/utils/VideoUtils.h"

#include <map>
#include <future>
#include <unordered_set>

extern "C" {
#include <libavcodec/bsf.h>
}

#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/FileUtils.h"
#include "acc/utils/ThreadPool.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace {
using namespace Acc;
constexpr int INVALID_VIDEO_STREAM_INDEX = -1;
constexpr int64_t INVALID_KEYFRAME_PTS = -1;
constexpr int VIDEO_DECODE_PARALLEL_NUM = 8;

ErrorCode OpenInputFile(const std::string& file, AVFormatContext*& formatCtx)
{
    if (avformat_open_input(&formatCtx, file.c_str(), nullptr, nullptr) != 0) {
        LogDebug << "Call avformat_open_input open file create format context failed, may caused by out of memory"
                    " or file is broken/invalid"
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        LogDebug << "Call avformat_find_stream_info find video stream information, please check the video file is"
                    "complete and contains a stream."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        avformat_close_input(&formatCtx);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    return SUCCESS;
}

ErrorCode CreateCodecContext(AVFormatContext& formatCtx, int videoStreamIndex, AVCodecContext*& codecCtx)
{
    AVCodecParameters* codecParameters = formatCtx.streams[videoStreamIndex]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecParameters->codec_id);
    if (!codec) {
        LogDebug << "Cannot find decoder, may caused by wrong codec_id, please cheack the raw video file."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }

    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        LogDebug << "Call avcodec_alloc_context3 malloc failed, please check system status.";
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    avcodec_parameters_to_context(codecCtx, codecParameters);
    codecCtx->thread_count = VIDEO_DECODE_PARALLEL_NUM;
    codecCtx->thread_type = FF_THREAD_FRAME;

    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        LogDebug << "Cannot open codec context, may caused by wrong codec param or out of memory."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        avcodec_free_context(&codecCtx);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    return SUCCESS;
}

bool DecodeFrames(AVFormatContext* formatCtx, AVCodecContext* codecCtx, int videoStreamIndex, int keyframeIdx,
                  VideoAuxInfo& videoAuxInfo, AVFrame* frame, AVPacket& packet,
                  std::map<int, AVFrame*>& yuvFrameResults)
{
    while (av_read_frame(formatCtx, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecCtx, &packet) != SUCCESS) {
                av_packet_unref(&packet);
                continue;
            }
            if (ReceiveVideoFrames(frame, codecCtx, keyframeIdx, &videoAuxInfo, yuvFrameResults) &&
                videoAuxInfo.IsDecodeDone(keyframeIdx)) {
                break;
            }
        }
        av_packet_unref(&packet);
    }

    // flush decoder
    if (!videoAuxInfo.IsDecodeDone(keyframeIdx)) {
        if (avcodec_send_packet(codecCtx, nullptr) == 0) {
            ReceiveVideoFrames(frame, codecCtx, keyframeIdx, &videoAuxInfo, yuvFrameResults);
        }
    }
    return !yuvFrameResults.empty();
}

bool ConvertYuvFramesToRgb(const std::map<int, AVFrame*>& yuvFrameResults, AVCodecContext& codecCtx,
                           std::map<int, AVFrame*>& results)
{
    ThreadPool& pool = ThreadPool::GetInstance();
    std::vector<std::future<std::pair<int, AVFrame*>>> futures;

    int width = codecCtx.width;
    int height = codecCtx.height;
    AVPixelFormat pixelFormat = codecCtx.pix_fmt;

    auto ConvertYuvToRgbTask = [](int frameIdx, AVFrame& yuvFrame, int width, int height, AVPixelFormat pixelFormat) {
        SwsContext* swsCtxThread = sws_getContext(width, height, pixelFormat, width, height, AV_PIX_FMT_RGB24,
                                                  SWS_BICUBIC, nullptr, nullptr, nullptr);
        if (!swsCtxThread) {
            LogDebug << "Create sws context failed, may caused by wrong param width, height or pixel format.";
            return std::make_pair(frameIdx, static_cast<AVFrame*>(nullptr));
        }
        AVFrame* rgbFrame = ConvertYuvToRgb(yuvFrame, swsCtxThread);
        sws_freeContext(swsCtxThread);
        return std::make_pair(frameIdx, rgbFrame);
    };
    for (auto& [frameIdx, yuvFramePtr] : yuvFrameResults) {
        futures.push_back(pool.Submit(
            [frameIdx, &yuvFramePtr, width, height, pixelFormat, &ConvertYuvToRgbTask]() -> std::pair<int, AVFrame*> {
                return ConvertYuvToRgbTask(frameIdx, *yuvFramePtr, width, height, pixelFormat);
            }));
    }
    bool isConvertSuccess = true;
    for (auto& future : futures) {
        auto [frameIdx, rgbFrame] = future.get();
        if (rgbFrame) {
            results[frameIdx] = rgbFrame;
        } else {
            LogDebug << "Frame " << frameIdx << " convert failed.";
            isConvertSuccess = false;
        }
    }
    return isConvertSuccess;
}

} // namespace

namespace Acc {

std::vector<int> VideoAuxInfo::GetTargetKeyframeIndices() const
{
    std::vector<int> targetKeyframeIndices;
    // pre-allocate memory to improve efficiency
    targetKeyframeIndices.reserve(keyframesTargetIndices.size());

    // keyframesAllPts and keyframesTargetIndices keys are intersected
    for (const auto& pair : keyframesTargetIndices) {
        targetKeyframeIndices.push_back(pair.first);
    }
    return targetKeyframeIndices;
}

int64_t VideoAuxInfo::KeyframeIdxToPts(int keyframeIdx) const
{
    auto it = keyframeIdxToPts.find(keyframeIdx);
    if (it != keyframeIdxToPts.end()) {
        return it->second;
    }
    LogDebug << "Keyframe index " << keyframeIdx << " not found in keyframeIdxToPts, please check construct func.";
    return INVALID_KEYFRAME_PTS;
}

int VideoAuxInfo::PtsToFrameIdx(const int64_t pts, int& frameIdx) const
{
    auto it = ptsToFrameIdx.find(pts);
    if (it != ptsToFrameIdx.end()) {
        frameIdx = it->second;
    } else {
        LogDebug << "Keyframe index does not exist in the pts to frame map, may calculated "
                    "wrong at ConstructVideoAuxInfo func."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    return SUCCESS;
}

bool VideoAuxInfo::ShouldConvert(const int64_t pts, int keyframeIdx) const
{
    auto it = keyframesAllPts.find(keyframeIdx);
    std::set<int64_t> framePts;
    if (it != keyframesAllPts.end()) {
        framePts = it->second;
    } else {
        LogDebug << "Keyframe index does not exist in the all keyframes pts set, may calculated "
                    "wrong at ConstructVideoAuxInfo func.";
        return false;
    }
    int frameIndex = 0;
    int ret = PtsToFrameIdx(pts, frameIndex);
    if (ret != SUCCESS) {
        LogDebug << "PtsToFrameIdx execute failed.";
        return false;
    }
    return targetIndices.find(frameIndex) != targetIndices.end() && framePts.find(pts) != framePts.end();
}

void VideoAuxInfo::UpdateDecodedNum(int keyframeIdx)
{
    std::unique_lock lock(decodedTargetIndicesNumMutex);
    decodedTargetIndicesNum[keyframeIdx]++;
}

bool VideoAuxInfo::IsDecodeDone(int keyframeIdx) const
{
    auto it = keyframesTargetIndices.find(keyframeIdx);
    size_t expectedFrameNum = 0;
    int decodedFrameNum = 0;
    if (it != keyframesTargetIndices.end()) {
        expectedFrameNum = it->second.size();
    } else {
        return false;
    }

    {
        std::shared_lock lock(decodedTargetIndicesNumMutex);
        auto it1 = decodedTargetIndicesNum.find(keyframeIdx);
        if (it1 != decodedTargetIndicesNum.end()) {
            decodedFrameNum = it1->second;
        } else {
            return false;
        }
    }
    return expectedFrameNum <= static_cast<size_t>(decodedFrameNum);
}

int FindVideoStream(AVFormatContext* formatCtx)
{
    if (formatCtx == nullptr) {
        LogDebug << "Find video stream failed, AVFormatContext is nullptr, may caused by previous step.";
        return INVALID_VIDEO_STREAM_INDEX;
    }
    for (uint32_t i = 0; i < formatCtx->nb_streams; i++) {
        if (!formatCtx->streams[i] || !formatCtx->streams[i]->codecpar) {
            LogDebug << "Find video stream failed, streams or codecpar is nullptr, may caused by previous step.";
            continue;
        }
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            return i;
        }
    }
    return INVALID_VIDEO_STREAM_INDEX;
}

ErrorCode GetFramesAndFPS(AVStream* videoStream, double& originFps, int64_t& totalFrames)
{
    if (videoStream == nullptr) {
        LogDebug << "Calculate video fps failed, AVStream is nullptr." << GetErrorInfo(ERR_INVALID_POINTER);
        return ERR_INVALID_POINTER;
    }
    totalFrames = videoStream->nb_frames;
    originFps = av_q2d(videoStream->avg_frame_rate);
    if (originFps > 0 && totalFrames > 0) {
        return SUCCESS;
    }

    if (originFps <= 0) {
        originFps = av_q2d(videoStream->r_frame_rate);
        if (originFps <= 0 && videoStream->duration > 0 && totalFrames > 0) {
            double durationSec = videoStream->duration * av_q2d(videoStream->time_base);
            if (durationSec > 0) {
                originFps = totalFrames / durationSec;
                return SUCCESS;
            }
        }
    }

    if (totalFrames <= 0) {
        double durationSec =
            (videoStream->duration != AV_NOPTS_VALUE) ? videoStream->duration * av_q2d(videoStream->time_base) : 0;
        if (durationSec > 0 && originFps > 0) {
            totalFrames = static_cast<int64_t>(durationSec * originFps);
            return SUCCESS;
        }
    }
    return ERR_FFMPEG_COMMON_FAILURE;
}

ErrorCode ConstructVideoAuxInfo(AVFormatContext* formatCtx, int videoStreamIdx, std::set<uint32_t> targetIndices,
                                VideoAuxInfo& videoAuxInfo)
{
    bool eof = false;
    auto packet = av_packet_alloc();
    if (!packet) {
        LogDebug << "Call av_packet_alloc malloc packet failed, please check system status.";
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    std::unordered_set<int64_t> ptsOfKeyFrames;
    std::vector<int64_t> ptsOfAllFrames;
    while (!eof) {
        // read each packet in a loop
        int ret = av_read_frame(formatCtx, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                eof = true;
            }
            break;
        }
        // if is video stream, insert into the timestamp set
        if (packet->stream_index == videoStreamIdx) {
            // if is key frame, insert into the keyframe timestamp set
            if ((static_cast<uint32_t>(packet->flags) & AV_PKT_FLAG_KEY) != 0) {
                ptsOfKeyFrames.insert(packet->pts);
            }
            ptsOfAllFrames.emplace_back(packet->pts);
        }
        av_packet_unref(packet);
    }
    av_packet_free(&packet);

    std::unordered_map<int, std::set<int64_t>> keyframesAllPts;
    std::unordered_map<int, std::set<int64_t>> keyframesTargetIndices;
    std::unordered_map<int, int> targetIdxToKeyframeIdx;
    std::unordered_map<int64_t, int> ptsToFrameIdx;
    std::unordered_map<int, int64_t> keyframeIdxToPts;
    std::sort(ptsOfAllFrames.begin(), ptsOfAllFrames.end());
    int keyFrameIdx = 0;
    for (size_t i = 0; i < ptsOfAllFrames.size(); ++i) {
        // i: frame index, pts: corresponding timestamp
        int64_t pts = ptsOfAllFrames[i];
        ptsToFrameIdx.insert(std::make_pair(pts, i));
        // find pts in key frame pts set, insert key frame id into id set
        if (ptsOfKeyFrames.find(pts) != ptsOfKeyFrames.end()) {
            keyFrameIdx = static_cast<int>(i);
            keyframeIdxToPts.insert(std::make_pair(i, pts));
        }
        keyframesAllPts[keyFrameIdx].insert(pts);
        // if current id equals to target id, record with the corresponding key frame
        if (targetIndices.find(i) != targetIndices.end()) {
            targetIdxToKeyframeIdx[i] = keyFrameIdx;
            keyframesTargetIndices[keyFrameIdx].insert(i);
        }
    }
    videoAuxInfo = VideoAuxInfo(std::move(keyframesAllPts), std::move(ptsToFrameIdx), std::move(keyframeIdxToPts),
        std::move(targetIndices), std::move(keyframesTargetIndices), std::move(targetIdxToKeyframeIdx));
    return SUCCESS;
}

bool ReceiveVideoFrames(AVFrame* frame, AVCodecContext* codecCtx, int keyframeIdx, VideoAuxInfo* videoAuxInfo,
                        std::map<int, AVFrame*>& results)
{
    if (frame == nullptr || codecCtx == nullptr || videoAuxInfo == nullptr) {
        LogDebug << "Parameter frame or codecCtx or videoAuxInfo is nullptr, please check.";
        return false;
    }
    bool res = false;
    while (true) {
        int ret = avcodec_receive_frame(codecCtx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret != SUCCESS) {
            res = false;
            break;
        }
        if (videoAuxInfo->ShouldConvert(frame->pts, keyframeIdx)) {
            videoAuxInfo->UpdateDecodedNum(keyframeIdx);
            auto rgbFrame = av_frame_clone(frame);
            if (!rgbFrame) {
                LogDebug << "Call av_frame_clone failed.";
                break;
            }
            int frameIdx = 0;
            int findRet = videoAuxInfo->PtsToFrameIdx(frame->pts, frameIdx);
            if (findRet != SUCCESS) {
                av_frame_free(&rgbFrame);
                break;
            }
            results.insert({frameIdx, rgbFrame});
            res = true;
        }
    }
    return res;
}

AVFrame* ConvertYuvToRgb(AVFrame& frame, SwsContext* swsCtx)
{
    AVFrame* rgbFrame = av_frame_alloc();
    if (rgbFrame == nullptr) {
        LogDebug << "Call av_frame_alloc failed, please check system status.";
        return nullptr;
    }
    rgbFrame->format = AV_PIX_FMT_RGB24;
    rgbFrame->width = frame.width;
    rgbFrame->height = frame.height;
    rgbFrame->pts = frame.pts;

    int ret = av_image_alloc(rgbFrame->data, rgbFrame->linesize, frame.width, frame.height, AV_PIX_FMT_RGB24, 1);
    if (ret < 0) {
        LogDebug << "Call av_image_alloc malloc buffer failed, please check system status.";
        av_frame_free(&rgbFrame);
        return nullptr;
    }

    // convert yuv420p to rgb24
    ret = sws_scale(swsCtx, frame.data, frame.linesize, 0, frame.height, rgbFrame->data, rgbFrame->linesize);
    if (ret < 0) {
        LogDebug << "Call sws_scale convert yuv to rgb frame failed.";
        av_frame_free(&rgbFrame);
        return nullptr;
    }
    return rgbFrame;
}

void FreeDecodeResource(AVFrame* frame, AVPacket* packet, AVCodecContext* codecCtx, AVFormatContext* formatCtx,
                        std::map<int, AVFrame*>& yuvFrameResults)
{
    for (auto& [frameIdx, yuvFrame] : yuvFrameResults) {
        av_frame_free(&yuvFrame);
    }
    yuvFrameResults.clear();

    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);
}

ErrorCode VideoDecodeSeek(const std::string& file, int keyframeIdx, VideoAuxInfo& videoAuxInfo,
                          std::map<int, AVFrame*>& results)
{
    AVFormatContext* formatCtx = nullptr;
    auto ret = OpenInputFile(file, formatCtx);
    if (ret != SUCCESS) {
        return ret;
    }
    int videoStreamIndex = FindVideoStream(formatCtx);
    if (videoStreamIndex == -1) {
        avformat_close_input(&formatCtx);
        LogDebug << "No video stream found in formatCtx, maybe the file is not contains video stream."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        return ERR_FFMPEG_COMMON_FAILURE;
    }

    AVCodecContext* codecCtx = nullptr;
    if (CreateCodecContext(*formatCtx, videoStreamIndex, codecCtx) != SUCCESS) {
        avformat_close_input(&formatCtx);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    if (!frame || !packet) {
        LogDebug << "Call av_packet_alloc or av_frame_alloc malloc failed, please check system status.";
        avformat_close_input(&formatCtx);
        avcodec_free_context(&codecCtx);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    int64_t targetPts = videoAuxInfo.KeyframeIdxToPts(keyframeIdx);
    // seek the target index
    if (av_seek_frame(formatCtx, videoStreamIndex, targetPts, AVSEEK_FLAG_BACKWARD) < 0) {
        LogDebug << "Cannot seek the to the target frame, may be the stream file is not support seek or "
                    "the given index is out of range."
                 << GetErrorInfo(ERR_FFMPEG_COMMON_FAILURE);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avformat_close_input(&formatCtx);
        avcodec_free_context(&codecCtx);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    avcodec_flush_buffers(codecCtx);
    std::map<int, AVFrame*> yuvFrameResults;
    if (!DecodeFrames(formatCtx, codecCtx, videoStreamIndex, keyframeIdx, videoAuxInfo, frame, *packet,
                      yuvFrameResults) ||
        !ConvertYuvFramesToRgb(yuvFrameResults, *codecCtx, results)) {
        FreeDecodeResource(frame, packet, codecCtx, formatCtx, yuvFrameResults);
        return ERR_FFMPEG_COMMON_FAILURE;
    }
    FreeDecodeResource(frame, packet, codecCtx, formatCtx, yuvFrameResults);
    return SUCCESS;
}
} // namespace Acc