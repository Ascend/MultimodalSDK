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
 * Description: Audio header file.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */

#ifndef AUDIO_H
#define AUDIO_H

#include <optional>
#include "acc/ErrorCode.h"
#include "acc/tensor/Tensor.h"

namespace Acc {

ErrorCode LoadAudioSingle(const std::string path, Tensor& result, int& originalSr,
                          std::optional<int> sr = std::nullopt);
/**
 * @brief Loads an audio file and optionally resamples it.
 * @param path: Path to the audio file.
 * @param result: Output tensor that stores the decoded (and resampled) mono audio data.
 * @param original_sr: Sample rate of the original audio file before resampling.
 * @param sr: Optional target sample rate. If not specified, the audio will not be resampled.
 * @return ErrorCode
 */
ErrorCode LoadAudioBatch(const std::vector<std::string> wavFiles, std::vector<Tensor>& results,
                         std::vector<int>& originalSrs, std::optional<int> sr = std::nullopt);
/**
 * @brief Loads multiple audio files and optionally resamples them.
 * @param wavFiles: List of paths to audio files.
 * @param results: Output tensors that store decoded (and resampled) mono audio data.
 * @param original_srs: Sample rates of the original audio files before resampling.
 * @param sr: Optional target sample rate. If not specified, the audio files will not be resampled.
 * @return ErrorCode
 */
} // namespace Acc
#endif // AUDIO_H
