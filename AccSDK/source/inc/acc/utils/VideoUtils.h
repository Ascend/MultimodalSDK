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
 * Description: Internal video utils header file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef VIDEO_UTILS_H
#define VIDEO_UTILS_H

#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <set>
#include <shared_mutex>
#include <mutex>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include "acc/ErrorCode.h"

namespace Acc {
// decode constraints
constexpr uint32_t MAX_STREAM_HEIGHT = 4096;   // Max video stream height
constexpr uint32_t MAX_STREAM_WIDTH = 4096;    // Max video stream width
constexpr uint32_t MIN_STREAM_HEIGHT = 480;    // Min video stream height
constexpr uint32_t MIN_STREAM_WIDTH = 480;     // Min video stream height

// cpu video decode aux info
class VideoAuxInfo {
public:
    // Default constructor
    VideoAuxInfo() = default;

    VideoAuxInfo(std::unordered_map<int, std::set<int64_t>> keyframesAllPts,
                 std::unordered_map<int64_t, int> ptsToFrameIdx, std::unordered_map<int, int64_t> keyframeIdxToPts,
                 std::set<uint32_t> targetIndices, std::unordered_map<int, std::set<int64_t>> keyframesTargetIndices,
                 std::unordered_map<int, int> targetIdxToKeyframeIdx)
        : keyframesAllPts(std::move(keyframesAllPts)),
          ptsToFrameIdx(std::move(ptsToFrameIdx)),
          keyframeIdxToPts(std::move(keyframeIdxToPts)),
          targetIndices(std::move(targetIndices)),
          keyframesTargetIndices(std::move(keyframesTargetIndices)),
          targetIdxToKeyframeIdx(std::move(targetIdxToKeyframeIdx))
    {
    }

    // Default destructor
    ~VideoAuxInfo() = default;

    // delete copy constructor
    VideoAuxInfo(const VideoAuxInfo&) = delete;

    // delete assignment operations
    VideoAuxInfo& operator=(const VideoAuxInfo&) = delete;

    // move constructor
    VideoAuxInfo(VideoAuxInfo&& other) noexcept
    {
        keyframesAllPts = std::move(other.keyframesAllPts);
        ptsToFrameIdx = std::move(other.ptsToFrameIdx);
        keyframeIdxToPts = std::move(other.keyframeIdxToPts);
        targetIndices = std::move(other.targetIndices);
        keyframesTargetIndices = std::move(other.keyframesTargetIndices);
        targetIdxToKeyframeIdx = std::move(other.targetIdxToKeyframeIdx);
        {
            std::unique_lock lock(other.decodedTargetIndicesNumMutex);
            decodedTargetIndicesNum = std::move(other.decodedTargetIndicesNum);
        }
    }

    VideoAuxInfo& operator=(VideoAuxInfo&& other) noexcept
    {
        if (this != &other) {
            keyframesAllPts = std::move(other.keyframesAllPts);
            ptsToFrameIdx = std::move(other.ptsToFrameIdx);
            keyframeIdxToPts = std::move(other.keyframeIdxToPts);
            targetIndices = std::move(other.targetIndices);
            keyframesTargetIndices = std::move(other.keyframesTargetIndices);
            targetIdxToKeyframeIdx = std::move(other.targetIdxToKeyframeIdx);
            {
                std::unique_lock lock_this(decodedTargetIndicesNumMutex, std::defer_lock);
                std::unique_lock lock_other(other.decodedTargetIndicesNumMutex, std::defer_lock);
                std::lock(lock_this, lock_other);
                decodedTargetIndicesNum = std::move(other.decodedTargetIndicesNum);
            }
        }
        return *this;
    }

    /**
     * @description: Get user specified indices corresponding keyframe indices.
     * @return: Corresponding keyframe indices
     */
    std::vector<int> GetTargetKeyframeIndices() const;

    /**
     * @description: Map keyframe index to time stamp.
     * @param: keyframe index
     * @return: corresponding time stamp
     */
    int64_t KeyframeIdxToPts(int keyframeIdx) const;

    /**
     * @description: Map time stamp to index.
     * @param pts: time stamp
     * @param frameIdx: frame id
     * @return: Error code
     */
    int PtsToFrameIdx(const int64_t pts, int& frameIdx) const;

    /**
     * @description: Judge the frame should convert or not.
     * @param pts: frame time stamp.
     * @param keyframeIdx: corresponding keyframe index
     * @return: Bool
     */
    bool ShouldConvert(const int64_t pts, int keyframeIdx) const;

    /**
     * @description: update decoded frames number
     * @param keyframeIdx: keyframe Index
     */
    void UpdateDecodedNum(int keyframeIdx);

    /**
     * @description: Judge the keyframe corresponding indices has decoded finished
     * @param keyframeIdx: keyframe index
     * @return: Bool
     */
    bool IsDecodeDone(int keyframeIdx) const;

private:
    // key: key frame index, value: all timestamps between current keyframe index and next keyframe
    std::unordered_map<int, std::set<int64_t>> keyframesAllPts;
    std::unordered_map<int64_t, int> ptsToFrameIdx;
    std::unordered_map<int, int64_t> keyframeIdxToPts;
    // The video frame id retained after video decoding, specified by user
    std::set<uint32_t> targetIndices;
    // key: key frame index, value: the key frame corresponding decoded target indices
    std::unordered_map<int, int> decodedTargetIndicesNum;
    // key: key frame index, value: the key frame corresponding target indices
    std::unordered_map<int, std::set<int64_t>> keyframesTargetIndices;
    // key: target index after video decode, value: corresponding key frame index
    std::unordered_map<int, int> targetIdxToKeyframeIdx;
    mutable std::shared_mutex decodedTargetIndicesNumMutex;
};

/**
 * @description: Find video stream index
 * @param formatCtx: AVFormatContext
 * @return: Int, video stream index, invalid stream index is -1
 */
int FindVideoStream(AVFormatContext* formatCtx);

/**
 * @description: Calculate video origin fps and total frames
 * @param videoStream: AV video stream pointer
 * @param originFps: output, origin fps
 * @param totalFrames: output, total frames
 * @return: Int, Error code
 */
ErrorCode GetFramesAndFPS(AVStream* videoStream, double& originFps, int64_t& totalFrames);

/**
 * @description: Construct video aux info for video decode
 * @param formatCtx: AVFormatContext
 * @param videoStreamIdx: Video stream index
 * @param targetIndices: user specified frame indices to decode
 * @param VideoAuxInfo: Video aux info for video decoding
 * @return: ErrorCode, Error code
 */
ErrorCode ConstructVideoAuxInfo(AVFormatContext* formatCtx, int videoStreamIdx, std::set<uint32_t> targetIndices,
                                VideoAuxInfo& videoAuxInfo);

/**
 * @description: Receive decoded video frames and construct frames to yuvToRgb
 * @param frame: decoded frame
 * @param codecCtx: AVCodecContext
 * @param keyframeIdx: Keyframe index
 * @param videoAuxInfo: Video aux info for decoding
 * @param results: decoded result
 * @return: Bool
 */
bool ReceiveVideoFrames(AVFrame* frame, AVCodecContext* codecCtx, int keyframeIdx, VideoAuxInfo* videoAuxInfo,
                        std::map<int, AVFrame*>& results);

/**
 * @description: Convert yuv frame to rgb
 * @param frame: AVFrame, to do convert color
 * @param swsCtx: SwsContext for convert color
 * @return: Rgb Frame
 */
AVFrame* ConvertYuvToRgb(AVFrame& frame, SwsContext* swsCtx);

/**
 * @description: Video decode for each thread
 * @param file: video mp4 file
 * @param keyframeIdx: corresponding keyframe index
 * @param videoAuxInfo: Video aux info for decoding
 * @param results: Decoded Rgb Frames
 * @return Int, Error Code
 */
ErrorCode VideoDecodeSeek(const std::string& file, int keyframeIdx, VideoAuxInfo& videoAuxInfo,
                          std::map<int, AVFrame*>& results);
} // namespace Acc
#endif