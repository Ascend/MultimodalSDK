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
 * Description: preprocess file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "PyPreprocess.h"

#include <memory>
#include <vector>

#include "Python.h"
#include "acc/ErrorCode.h"
#include "acc/fusion_operators/FusionOperators.h"

namespace PyAcc
{

std::vector<Tensor> Qwen2VLProcessor::Preprocess(const std::vector<Image>& pyImages, const std::vector<float>& mean,
                                                 const std::vector<float>& std, int resizeW, int resizeH)
{
    if (pyImages.empty())
    {
        throw std::runtime_error("Images is empty. This is likely due to a corrupt image.");
    }

    std::vector<std::shared_ptr<Acc::Image>> internalImages;
    internalImages.reserve(pyImages.size());
    for (const auto& pyImage : pyImages)
    {
        auto internalImage = pyImage.GetImagePtr();
        if (!internalImage)
        {
            throw std::runtime_error("Failed to retrieve image data. This is likely due to a corrupt image.");
        }
        internalImages.push_back(internalImage);
    }

    Acc::QwenPreprocessConfig config{mean, std, resizeW, resizeH};

    std::vector<Acc::Tensor> accTensors;
    Acc::ErrorCode ret = Acc::FusionOperator::Qwen2VLImagePreprocess<Acc::Image>(internalImages, config, accTensors);
    if (ret != Acc::SUCCESS || accTensors.empty())
    {
        throw std::runtime_error("Failed to preprocess image data. Please see above log for detail.");
    }

    std::vector<Tensor> pyTensors;
    pyTensors.reserve(accTensors.size());
    for (auto& accTensor : accTensors)
    {
        Tensor pyTensor;
        pyTensor.SetTensor(accTensor);
        pyTensors.push_back(std::move(pyTensor));
    }

    return pyTensors;
}

std::vector<Tensor> Qwen2VLProcessor::PreprocessTensor(const std::vector<Tensor>& pyTensors,
                                                       const std::vector<float>& mean, const std::vector<float>& std,
                                                       int resizeW, int resizeH)
{
    if (pyTensors.empty())
    {
        throw std::runtime_error("Tensors is empty. This is likely due to a corrupt tensor.");
    }

    std::vector<std::shared_ptr<Acc::Tensor>> internalTensors;
    internalTensors.reserve(pyTensors.size());
    for (const auto& pyTensor : pyTensors)
    {
        auto internalTensor = pyTensor.GetTensorPtr();
        if (!internalTensor)
        {
            throw std::runtime_error("Failed to retrieve tensor data. This is likely due to a corrupt tensor.");
        }
        internalTensors.push_back(internalTensor);
    }

    Acc::QwenPreprocessConfig config{mean, std, resizeW, resizeH};

    std::vector<Acc::Tensor> accTensors;
    Acc::ErrorCode ret = Acc::FusionOperator::Qwen2VLImagePreprocess<Acc::Tensor>(internalTensors, config, accTensors);
    if (ret != Acc::SUCCESS || accTensors.empty())
    {
        throw std::runtime_error("Failed to preprocess tensor data. Please see above log for detail.");
    }

    std::vector<Tensor> pyTensors_;
    pyTensors_.reserve(accTensors.size());
    for (auto& accTensor : accTensors)
    {
        Tensor pyTensor;
        pyTensor.SetTensor(accTensor);
        pyTensors_.push_back(std::move(pyTensor));
    }

    return pyTensors_;
}

}  // namespace PyAcc
