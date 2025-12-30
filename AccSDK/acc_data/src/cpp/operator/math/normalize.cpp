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
 * @Description:
 * @Version: 1.0
 * @Date: 2025-2-14 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-14 9:00:00
 */

#include "normalize.h"

#include <arm_neon.h>

#include "operator/op_factory.h"
#include "common/balance.h"
#include "tensor/tensor_image.h"
#include "common/tracer.h"

namespace acclib {
namespace accdata {

using OutputType = float;  // Now assume the output datatype is float.

AccDataErrorCode Normalize::Run(Workspace& ws)
{
    TRACE_BEGIN(norm)
    auto errCode = AccDataErrorCode::H_OK;
    auto &pool = ws.GetThreadPool();
    auto &inputList = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);

    if (inputList.IsEmpty() || !inputList.IsValid()) {
        ACCDATA_ERROR("Illegal input.");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }

    auto &outputList = ws.GetOutput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);

    errCode = Setup(ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the normalize operator",
                                   errCode);

    errCode = outputList.Resize<OutputType>(inputList.Shape());
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to resize.", errCode);

    int numTensors = inputList.NumTensors();
    for (int i = 0; i < numTensors; ++i) {
        auto& input = inputList[i];
        auto& output = outputList[i];
        errCode = AddTask(pool, input, output);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to add task.", errCode);
    }
    errCode = pool.RunAll();
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to run the normalize operator.", errCode);

    errCode = outputList.SetLayout(TensorLayoutView(inputList.Tensors()));
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set layout.", errCode);
    TRACE_END(norm)
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Normalize::Setup(Workspace& ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto& spec = GetSpec();
    if (ws.NumOutput() != spec.NumOutput()) {
        ACCDATA_ERROR("The number of outputs is inconsistent.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }

    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Input out of range during fused operation execution.", errCode);
    if (input.NumTensors() < 1) {
        ACCDATA_ERROR("The number of input is empty.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    errCode = mInputMeta.Setup(input[0]);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the input meta arguments.",
                                   errCode);

    errCode = mNormalizeArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the normalize arguments.",
                                   errCode);
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode Normalize::AddTask(ThreadPool& pool, const Tensor& input, Tensor& output)
{
    switch (input.DataType()) {
        case TensorDataType::FP32:
            return AddTask<float, OutputType>(pool, input, output);
        default:
            ACCDATA_ERROR("The datatype of Normalize input should be float.");
            return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
}

template <typename InputType, typename OutputType>
AccDataErrorCode Normalize::AddTask(ThreadPool& pool, const Tensor& input, Tensor& output)
{
    switch (input.Layout()) {
        case TensorLayout::NHWC:
            return AddTask<InputType, OutputType, TensorLayout::NHWC>(pool, input, output);
        case TensorLayout::NCHW:
            return AddTask<InputType, OutputType, TensorLayout::NCHW>(pool, input, output);
        default:
            ACCDATA_ERROR("Unsupported layout '" << input.Layout() << "'.");
            return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
}

template <typename InputType, typename OutputType, TensorLayout Layout>
AccDataErrorCode Normalize::AddTask(ThreadPool& pool, const Tensor& input, Tensor& output)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto shape = input.Shape();
    auto numSamples = mInputMeta.NumSamples();
    uint64_t numThreads = pool.NumThreads();
    OperatorParam param;
    SetupParam(param);

    auto* in = input.RawDataPtr<InputType>();
    auto* out = output.RawDataPtr<OutputType>();
    Balance::Task range = {0, 0};
    for (uint64_t t = 0; t < numThreads; ++t) {
        errCode = Balance::Assign(numSamples, numThreads, t, range);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.", errCode);
        if (range.begin >= range.end) {
            break;
        }
        param.begin = range.begin;
        param.end = range.end;
        auto task = [this, in, out, param](int id, AccDataErrorCode &errCode) {
            RunTask<InputType, OutputType, Layout>(in, param, out, errCode);
        };
        pool.AddTask(task);
    }
    return AccDataErrorCode::H_OK;
}

void Normalize::SetupParam(OperatorParam &param)
{
    param.height = mInputMeta.Height();
    param.width = mInputMeta.Width();
    param.channel = mInputMeta.NumChannels();
    param.begin = 0;
    param.end = 0;
}

template <typename InputType, typename OutputType, TensorLayout Layout>
void Normalize::RunTask(const InputType* input, const OperatorParam& param, OutputType* output,
                        AccDataErrorCode &errCode)
{
    if constexpr (Layout == TensorLayout::NHWC) {
        RunHWC(input, param, output);
        errCode = AccDataErrorCode::H_OK;
    } else if constexpr (Layout == TensorLayout::NCHW) {
        RunCHW(input, param, output);
        errCode = AccDataErrorCode::H_OK;
    } else {
        ACCDATA_ERROR("Unsupported layout '" << Layout << "'.");
        errCode = AccDataErrorCode::H_SINGLEOP_ERROR;
    }
    return;
}

template <typename InputType, typename OutputType>
void Normalize::RunHWC(const InputType* input, const OperatorParam& param, OutputType* output)
{
    auto& mean = mNormalizeArgs.Mean();
    auto& scale = mNormalizeArgs.Scale();
    auto begin = param.begin * param.height * param.width;
    auto end = param.end * param.height * param.width;
    constexpr int channelRed = 0;
    constexpr int channelGreen = 1;
    constexpr int channelBlue = 2;
    for (uint64_t i = begin; i < end; ++i) {
        output[i * param.channel] = (input[i * param.channel] - mean[channelRed]) * scale[channelRed];
        output[i * param.channel + channelGreen] =
            (input[i * param.channel + channelGreen] - mean[channelGreen]) * scale[channelGreen];
        output[i * param.channel + channelBlue] =
            (input[i * param.channel + channelBlue] - mean[channelBlue]) * scale[channelBlue];
    }
    return;
}

template <typename InputType, typename OutputType>
void Normalize::RunCHW(const InputType* input, const OperatorParam& param, OutputType* output)
{
    auto& mean = mNormalizeArgs.Mean();
    auto& scale = mNormalizeArgs.Scale();
    auto begin = param.begin * param.channel;
    auto end = param.end * param.channel;
    auto resolution = param.height * param.width;

    for (uint64_t i = begin; i < end; ++i) {
        uint32_t channel = i % 3;
        uint64_t offset = i * resolution;
        float32x4_t meanValue = vdupq_n_f32(mean[channel]);
        float32x4_t scaleValue = vdupq_n_f32(scale[channel]);
        uint64_t j = 0;
        for (; j < resolution - 3; j += 4) {    // 128位寄存器可以放4个float32,数据不足四个时单独处理
            float32x4_t datas = vld1q_f32(&input[offset + j]);
            datas = vsubq_f32(datas, meanValue);
            datas = vmulq_f32(datas, scaleValue);
            vst1q_f32(&output[offset + j], datas);
        }
        for (; j < resolution; ++j) {
            output[offset + j] = (input[offset + j] - mean[channel]) * scale[channel];
        }
    }
    return;
}

ACCDATA_REGISTER_OPERATOR(Normalize, acclib::accdata::Normalize);

}  // namespace accdata
}  // namespace acclib
