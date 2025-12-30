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

#include "to_tensor.h"

#include <atomic>

#include "operator/op_factory.h"
#include "common/balance.h"
#include "common/tracer.h"

namespace acclib {
namespace accdata {

AccDataErrorCode ToTensor::Run(Workspace &ws)
{
    TRACE_BEGIN(to_tensor)
    auto errCode = AccDataErrorCode::H_OK;
    ACCDATA_DEBUG("ToTensor run.");
    errCode = Setup(ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the to tensor operator.",
                                   errCode);

    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Input out of range during totensor operation execution.", errCode);

    auto &output = ws.GetOutput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Output out of range during totensor operation execution.", errCode);

    if (input.NumTensors() > output.NumTensors()) {
        ACCDATA_ERROR("The number of input tensors should not exceed that of output.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }

    TensorListShape outputShape;
    errCode = GetOutputShape(input, outputShape);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to get output shape.", errCode);

    errCode = output.Resize<float>(outputShape);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to resize.", errCode);

    auto &pool = ws.GetThreadPool();
    int numTensors = input.NumTensors();
    for (int i = 0; i < numTensors; ++i) {
        auto &in = input[i];
        auto &out = output[i];
        errCode = AddTask<uint8_t, float>(pool, in, out);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to add task.", errCode);
    }
    errCode = pool.RunAll();
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to run task.", errCode);

    output.SetLayout(mToTensorArgs.Layout());
    TRACE_END(to_tensor)

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ToTensor::Setup(Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto &spec = GetSpec();
    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Input out of range during totensor operation setup.", errCode);

    if (ws.NumOutput() != spec.NumOutput()) {
        ACCDATA_ERROR("The number of outputs should be consistent.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }

    errCode = mToTensorArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the to tensor arguments.",
                                   errCode);
    errCode = mInputMeta.Setup(input[0]);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the input meta.", errCode);

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ToTensor::GetOutputShape(const TensorList &input, TensorListShape &outputShape)
{
    auto numSamples = static_cast<size_t>(mInputMeta.NumSamples());
    auto height = static_cast<size_t>(mInputMeta.Height());
    auto width = static_cast<size_t>(mInputMeta.Width());
    auto numChannels = static_cast<size_t>(mInputMeta.NumChannels());

    switch (mToTensorArgs.Layout()) {
        case TensorLayout::NHWC:
            outputShape = TensorListShape(input.NumTensors(), { numSamples, height, width, numChannels });
            return AccDataErrorCode::H_OK;
        case TensorLayout::NCHW:
            outputShape = TensorListShape(input.NumTensors(), { numSamples, numChannels, height, width });
            return AccDataErrorCode::H_OK;
        default:
            ACCDATA_ERROR("Unsupported layout '" << input[0].Layout() << "'.");
            return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
}

template <typename InputType, typename OutputType>
AccDataErrorCode ToTensor::AddTask(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    // 输入输出布局相同，无需关心是NHWC或NCHW，只要按照连续空间访问所有数据
    if (ACCDATA_UNLIKELY(mToTensorArgs.IsSameLayout())) {
        return AddTaskSameLayout<InputType, OutputType>(pool, input, output);
    }

    if (input.Layout() == TensorLayout::NHWC) {
        return AddTaskInner<InputType, OutputType, TensorLayout::NHWC>(pool, input, output);
    } else if (input.Layout() == TensorLayout::NCHW) {
        return AddTaskInner<InputType, OutputType, TensorLayout::NCHW>(pool, input, output);
    } else {
        ACCDATA_ERROR("Unsupported layout '" << input.Layout() << "'.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
}

template <typename InputType, typename OutputType>
AccDataErrorCode ToTensor::AddTaskSameLayout(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    int numThreads = pool.NumThreads();
    int64_t numElm = NumElements(input.Shape());
    auto *in = input.RawDataPtr<InputType>();
    auto *out = output.RawDataPtr<OutputType>();
    auto mul = mToTensorArgs.Mul();
    Balance::Task range = {0, 0};
    for (int i = 0; i < numThreads; ++i) {
        auto errCode = Balance::Assign(numElm, numThreads, i, range);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.", errCode);
        if (range.begin >= range.end) {
            break;
        }
        auto task = [this, in, out, range, mul](int id, AccDataErrorCode &errCode = AccDataErrorCode::H_OK) {
            for (int64_t j = range.begin; j < range.end; ++j) {
                out[j] = in[j] * mul;
            }
        };
        pool.AddTask(task);
    }
    return AccDataErrorCode::H_OK;
}

OperatorParam ToTensor::SetupParam()
{
    OperatorParam param;
    param.height = mInputMeta.Height();
    param.width = mInputMeta.Width();
    param.channel = mInputMeta.NumChannels();
    param.begin = 0;
    param.end = 0;
    return param;
}

template <typename InputType, typename OutputType, TensorLayout Layout>
AccDataErrorCode ToTensor::AddTaskInner(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    auto shape = input.Shape();
    int numThreads = pool.NumThreads();
    OperatorParam param = SetupParam();

    auto *in = input.RawDataPtr<InputType>();
    auto *out = output.RawDataPtr<OutputType>();
    auto numSamples = mInputMeta.NumSamples();
    Balance::Task range = {0, 0};

    for (int t = 0; t < numThreads; ++t) {
        auto errCode = Balance::Assign(numSamples, numThreads, t, range);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.k", errCode);
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

template <typename InputType, typename OutputType, TensorLayout Layout>
void ToTensor::RunTask(const InputType *input, const OperatorParam param, OutputType *output, AccDataErrorCode &errCode)
{
    if constexpr (Layout == TensorLayout::NHWC) {
        Trans2NCHW(input, param, output);
        errCode = AccDataErrorCode::H_OK;
    } else if constexpr (Layout == TensorLayout::NCHW) {
        Trans2NHWC(input, param, output);
        errCode = AccDataErrorCode::H_OK;
    } else {
        ACCDATA_ERROR("Unsupported layout '" << Layout << "'.");
        errCode = AccDataErrorCode::H_SINGLEOP_ERROR;
    }
}

template <typename InputType, typename OutputType>
void ToTensor::Trans2NCHW(const InputType *input, const OperatorParam &param, OutputType *output)
{
    auto resolution = param.height * param.width;
    auto hwc = resolution * param.channel;
    auto mul = mToTensorArgs.Mul();

    for (auto i = param.begin; i < param.end; ++i) {
        auto offset = i * hwc;
        for (uint64_t j = 0; j < resolution; ++j) {
            output[offset + RGB_CHANNEL_RED * resolution + j] =
                input[offset + RGB_CHANNELS * j + RGB_CHANNEL_RED] * mul;
            output[offset + RGB_CHANNEL_GREEN * resolution + j] =
                input[offset + RGB_CHANNELS * j + RGB_CHANNEL_GREEN] * mul;
            output[offset + RGB_CHANNEL_BLUE * resolution + j] =
                input[offset + RGB_CHANNELS * j + RGB_CHANNEL_BLUE] * mul;
        }
    }
}

template <typename InputType, typename OutputType>
void ToTensor::Trans2NHWC(const InputType *input, const OperatorParam &param, OutputType *output)
{
    auto resolution = param.height * param.width;
    auto hwc = resolution * param.channel;
    auto mul = mToTensorArgs.Mul();

    for (auto i = param.begin; i < param.end; ++i) {
        auto offset = i * hwc;
        for (uint64_t j = 0; j < resolution; ++j) {
            output[offset + RGB_CHANNELS * j + RGB_CHANNEL_RED] =
                input[offset + RGB_CHANNEL_RED * resolution + j] * mul;
            output[offset + RGB_CHANNELS * j + RGB_CHANNEL_GREEN] =
                input[offset + RGB_CHANNEL_GREEN * resolution + j] * mul;
            output[offset + RGB_CHANNELS * j + RGB_CHANNEL_BLUE] =
                input[offset + RGB_CHANNEL_BLUE * resolution + j] * mul;
        }
    }
}

ACCDATA_REGISTER_OPERATOR(ToTensor, acclib::accdata::ToTensor);
} // namespace accdata
} // namespace acclib
