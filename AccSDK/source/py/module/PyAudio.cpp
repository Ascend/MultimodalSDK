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
 * Description: audio file for python.
 * Author: ACC SDK
 * Create: 2026
 * History: NA
 */
#include "PyAudio.h"
#include "PyUtil.h"
#include "acc/audio/Audio.h"
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"

namespace {
using namespace PyAcc;

void load_audio_impl(const std::string& path, Tensor& dst, int& originalSr, std::optional<int> sr)
{
    Acc::Tensor tensor;

    Acc::ErrorCode ret = LoadAudioSingle(path.c_str(), tensor, originalSr, sr);
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error(std::string("LoadAudio failed"));
    }
    dst.SetTensor(tensor);
}

void load_audio_batch_impl(std::vector<std::string> wavFiles, std::vector<Tensor>& dst, std::vector<int>& originalSrs,
                           std::optional<int> sr)
{
    std::vector<Acc::Tensor> tensors;

    Acc::ErrorCode ret = LoadAudioBatch(wavFiles, tensors, originalSrs, sr);
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error("LoadAudioBatch failed");
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        dst[i].SetTensor(tensors[i]);
    }
}

} // namespace

namespace PyAcc {

int load_audio(const std::string& path, Tensor& dst)
{
    int originalSr = 0;
    load_audio_impl(path, dst, originalSr, std::nullopt);
    return originalSr;
}

int load_audio(const std::string& path, Tensor& dst, int sr)
{
    int originalSr = 0;
    load_audio_impl(path, dst, originalSr, sr);
    return sr;
}

std::vector<int> load_audio_batch(std::vector<std::string> wavFiles, std::vector<Tensor>& dst)
{
    std::vector<int> originalSrs = {};
    load_audio_batch_impl(wavFiles, dst, originalSrs, std::nullopt);
    return originalSrs;
}

std::vector<int> load_audio_batch(std::vector<std::string> wavFiles, std::vector<Tensor>& dst, int sr)
{
    std::vector<int> originalSrs = {};
    load_audio_batch_impl(wavFiles, dst, originalSrs, sr);
    std::vector<int> srs(originalSrs.size(), sr);
    return srs;
}

} // namespace PyAcc