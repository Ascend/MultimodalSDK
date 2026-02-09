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
 * Description: audio head file for python.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#ifndef PYAUDIO_H
#define PYAUDIO_H

#include <string>
#include <vector>
#include "PyTensor.h"

namespace PyAcc {

struct LoadAudioResult {
    Tensor tensor;
    int original_sr;
};

/**
 * @brief Loads a single audio file into a tensor and optionally resamples it.
 * @param path: Path to the input audio file.
 * @param dst: Tensor to store the decoded (and optionally resampled) audio.
 * @param sr: Optional target sample rate. If not specified, the original sample rate is used.
 * @return original sample rate of the audio
 */
int load_audio(const std::string& path, Tensor& dst);
int load_audio(const std::string& path, Tensor& dst, int sr);

/**
 * @brief Loads multiple audio files into tensors and optionally resamples them.
 * @param wavFiles: List of audio file paths to load.
 * @param dst: Vector of tensors to store the decoded (and optionally resampled) audios.
 * @param sr: Optional target sample rate. If not specified, the original sample rates are used.
 * @return Vector of sample rates corresponding to each loaded audio
 */
std::vector<int> load_audio_batch(std::vector<std::string> wavFiles, std::vector<Tensor>& dst);
std::vector<int> load_audio_batch(std::vector<std::string> wavFiles, std::vector<Tensor>& dst, int sr);

} // namespace PyAcc
#endif // PYAUDIO_H