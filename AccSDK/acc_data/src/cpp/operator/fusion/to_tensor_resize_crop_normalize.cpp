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
 * @Date: 2025-2-20 14:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-20 14:00:00
 */

#include "to_tensor_resize_crop_normalize.h"

#include "common/balance.h"
#include "common/tracer.h"
#include "operator/op_factory.h"

namespace acclib {
namespace accdata {

AccDataErrorCode ToTensorResizeCropNormalize::Run(Workspace &ws)
{
    TRACE_BEGIN(fusion_to_tensor_resize_crop_norm)
    auto errCode = AccDataErrorCode::H_OK;

    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Input out of range during fused operation execution.", errCode);

    if (input.IsEmpty() || !input.IsValid()) {
        ACCDATA_ERROR("Illegal input.");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }

    errCode = Setup(ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to set up the to tensor resize crop normalize operator.", errCode);

    auto &output = ws.GetOutput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Output out of range during fused operation execution.", errCode);

    TensorListShape outputShape;
    errCode = GetOutputShape(input, outputShape);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to get output shape.", errCode);

    errCode = output.Resize<ResultType>(outputShape);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to resize.", errCode);

    auto &pool = ws.GetThreadPool();
    int numTensors = input.NumTensors();

    for (int i = 0; i < numTensors; ++i) {
        auto &in = input[i];
        auto &out = output[i];
        errCode = ClassifyTask<uint8_t, ResultType>(pool, in, out);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                       "Failed to classify task.", errCode);
    }

    auto resultLayout = mToTensorArgs.Layout();
    output.SetLayout(resultLayout);
    TRACE_END(fusion_to_tensor_resize_crop_norm)

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ToTensorResizeCropNormalize::Setup(Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto &spec = GetSpec();
    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range during input setup.", errCode);

    if (ws.NumOutput() != spec.NumOutput() || input.NumTensors() < 1) {
        ACCDATA_ERROR("The number of outputs should be consistent and input of ToTensor should not be empty.");
        return AccDataErrorCode::H_FUSIONOP_ERROR;
    }

    errCode = mToTensorArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to set up the to tensor arguments.", errCode);
    errCode = mInputMeta.Setup(input[0]);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to set up the input meta.", errCode);
    errCode = mCropArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to set up the crop arguments.", errCode);
    errCode = mResizeArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to set up the resize arguments.", errCode);

    if (mResizeArgs.Mode() != InterpMode::BILINEAR) {
        ACCDATA_ERROR("Unsupported interpolation mode " << mResizeArgs.Mode() << " for fused op.");
        return AccDataErrorCode::H_FUSIONOP_ERROR;
    }
    
    if (mCropArgs.Width() > mResizeArgs.Width() || mCropArgs.Height() > mResizeArgs.Height()) {
        ACCDATA_ERROR("Crop size can not be greater than resize size!!");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    errCode = mNormArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Failed to set up the normalize arguments.", errCode);

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ToTensorResizeCropNormalize::GetOutputShape(const TensorList &input, TensorListShape &outputShape)
{
    auto numSamples = static_cast<size_t>(mInputMeta.NumSamples());
    auto cropHeight = static_cast<size_t>(mCropArgs.Height());
    auto cropWidth = static_cast<size_t>(mCropArgs.Width());
    auto numChannels = static_cast<size_t>(mInputMeta.NumChannels());

    switch (mToTensorArgs.Layout()) {
        case TensorLayout::NHWC:
            outputShape = TensorListShape(input.NumTensors(), { numSamples, cropHeight, cropWidth, numChannels });
            return AccDataErrorCode::H_OK;
        case TensorLayout::NCHW:
            outputShape = TensorListShape(input.NumTensors(), { numSamples, numChannels, cropHeight, cropWidth });
            return AccDataErrorCode::H_OK;
        default:
            ACCDATA_ERROR("Unsupported layout '" << input[0].Layout() << "'.");
            return AccDataErrorCode::H_FUSIONOP_ERROR;
    }
}

OperatorParam ToTensorResizeCropNormalize::SetupParam()
{
    OperatorParam param;
    param.height = mInputMeta.Height();
    param.width = mInputMeta.Width();
    param.channel = mInputMeta.NumChannels();
    param.begin = 0;
    param.end = 0;
    param.cropOffsetX = mCropArgs.Left(mResizeArgs.Width());
    param.cropOffsetY = mCropArgs.Top(mResizeArgs.Height());

    return param;
}

template <typename InputType, typename OutputType>
AccDataErrorCode ToTensorResizeCropNormalize::ClassifyTask(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    auto inLayout = input.Layout();
    auto resultLayout = mToTensorArgs.Layout();

    switch (inLayout) {
        case TensorLayout::NHWC:
            if (resultLayout == TensorLayout::NCHW) {
                return AddTask<InputType, OutputType, TensorLayout::NHWC, TensorLayout::NCHW>(pool, input, output);
            } else if (resultLayout == TensorLayout::NHWC) {
                return AddTask<InputType, OutputType, TensorLayout::NHWC, TensorLayout::NHWC>(pool, input, output);
            } else {
                ACCDATA_ERROR("Unsupported result layout " << resultLayout);
                return AccDataErrorCode::H_FUSIONOP_ERROR;
            }
        case TensorLayout::NCHW:
            if (resultLayout == TensorLayout::NCHW) {
                return AddTask<InputType, OutputType, TensorLayout::NCHW, TensorLayout::NCHW>(pool, input, output);
            } else if (resultLayout == TensorLayout::NHWC) {
                return AddTask<InputType, OutputType, TensorLayout::NCHW, TensorLayout::NHWC>(pool, input, output);
            } else {
                ACCDATA_ERROR("Unsupported result layout " << resultLayout);
                return AccDataErrorCode::H_FUSIONOP_ERROR;
            }
        default:
            ACCDATA_ERROR("Unsupported input layout " << inLayout);
            return AccDataErrorCode::H_FUSIONOP_ERROR;
    }
}

template <typename InputType, typename OutputType, TensorLayout InLayout, TensorLayout OutLayout>
AccDataErrorCode ToTensorResizeCropNormalize::AddTask(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    int numThreads = pool.NumThreads();
    auto numberSamples = mInputMeta.NumSamples();
    OperatorParam param = SetupParam();

    auto *in = input.RawDataPtr<InputType>();
    auto *out = output.RawDataPtr<OutputType>();

    for (int t = 0; t < numThreads; ++t) {
        Balance::Task range = {0, 0};
        AccDataErrorCode errCode = Balance::Assign(numberSamples, numThreads, t, range);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.", errCode);

        if (range.begin >= range.end) {
            break;
        }

        param.begin = range.begin;
        param.end = range.end;

        // use reference for mem to release memory just once in the destructor without worrying exception
        auto task = [this, in, out, param](int id, AccDataErrorCode &errCode) {
            RunTask<InputType, OutputType, InLayout, OutLayout>(in, out, param, errCode);
        };
        pool.AddTask(task);
    }
    return pool.RunAll();
}

template <typename InputType, typename OutputType, TensorLayout InLayout, TensorLayout OutLayout>
void ToTensorResizeCropNormalize::RunTask(const InputType *input, OutputType *output, const OperatorParam &param,
                                          AccDataErrorCode &errCode)
{
    if constexpr (InLayout == TensorLayout::NHWC && OutLayout == TensorLayout::NCHW) {
        Kernel2NCHW<InputType, OutputType>(input, output, param);
        errCode = AccDataErrorCode::H_OK;
    } else if constexpr (InLayout == TensorLayout::NHWC && OutLayout == TensorLayout::NHWC) {
        ACCDATA_ERROR("Unimplemented NHWC to NHWC.");
        errCode = AccDataErrorCode::H_FUSIONOP_ERROR;
    } else if constexpr (InLayout == TensorLayout::NCHW && OutLayout == TensorLayout::NCHW) {
        ACCDATA_ERROR("Unimplemented NCHW to NCHW.");
        errCode = AccDataErrorCode::H_FUSIONOP_ERROR;
    } else if constexpr (InLayout == TensorLayout::NCHW && OutLayout == TensorLayout::NHWC) {
        ACCDATA_ERROR("Unimplemented NCHW to NHWC.");
        errCode = AccDataErrorCode::H_FUSIONOP_ERROR;
    } else {
        ACCDATA_ERROR("Unsupported layout input '" << InLayout << " result " << OutLayout << "'.");
        errCode = AccDataErrorCode::H_FUSIONOP_ERROR;
    }
}

template <typename InputType, typename OutputType>
void ToTensorResizeCropNormalize::Kernel2NCHW(const InputType *input, OutputType *output, const OperatorParam &param)
{
    auto ch = mCropArgs.Height();
    auto cw = mCropArgs.Width();
    auto hwcOrigin = param.height * param.width * param.channel;
    auto hwcResult = ch * cw * param.channel;
    auto resizeH = mResizeArgs.Height();
    auto resizeW = mResizeArgs.Width();
    auto cropH = mCropArgs.Height();
    auto cropW = mCropArgs.Width();

    auto scaleX = (OutputType *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(resizeW, sizeof(OutputType)));
    auto scaleY = (OutputType *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(resizeH, sizeof(OutputType)));
    auto depX = (int *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(resizeW, sizeof(int)));
    auto depY = (int *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(resizeH, sizeof(int)));
    // Allocate space for f(x,y0) and f(x, y1) in 3 channels, thus 2 * 3 = 6
    auto space = (OutputType *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(cropW * 6, sizeof(OutputType)));

    // precompute coefficient
    auto widthScale = static_cast<OutputType>(mInputMeta.Width()) / static_cast<OutputType>(resizeW);
    auto heightScale = static_cast<OutputType>(mInputMeta.Height()) / static_cast<OutputType>(resizeH);

    for (int64_t x = param.cropOffsetX; x < param.cropOffsetX + cropW; ++x) {
        OutputType fdx = std::max((x + 0.5) * widthScale - 0.5, 0.0);
        int idx = depX[x] = static_cast<int>(fdx);
        scaleX[x] = 1.0 + idx - fdx;
    }
    for (int64_t y = param.cropOffsetY; y < param.cropOffsetY + cropH; ++y) {
        OutputType fdy = std::max((y + 0.5) * heightScale - 0.5, 0.0);
        int idx = depY[y] = static_cast<int>(fdy);
        scaleY[y] = 1.0 + idx - fdy;
    }

    auto c00 = (OutputType *)space;  // f(x, y0) for channel 1
    auto c01 = c00 + cw;          // f(x, y1) for channel 1
    auto c02 = c01 + cw;          // f(x, y0) for channel 2
    auto c10 = c02 + cw;          // f(x, y1) for channel 2
    auto c11 = c10 + cw;          // f(x, y0) for channel 3
    auto c12 = c11 + cw;          // f(x, y1) for channel 3

    for (auto sample = param.begin; sample < param.end; ++sample) {
        auto src = input + sample * hwcOrigin;
        auto dst = output + sample * hwcResult;
        int oy = -2;

        for (auto y = param.cropOffsetY; y < param.cropOffsetY + ch; ++y) {
            auto y0 = depY[y];
            auto y1 = std::min(y0 + 1, (int)(param.height - 1));
            auto sy0 = scaleY[y];
            auto sy1 = 1.0f - sy0;

            if (y0 - oy >= RGB_CHANNEL_BLUE) {
                Compute3ChannelLine<InputType, OutputType>(src + y0 * param.width * param.channel, c00, c01, c02,
                                                           param.width, cw, depX + param.cropOffsetX,
                                                           scaleX + param.cropOffsetX, mToTensorArgs.Mul());
                Compute3ChannelLine<InputType, OutputType>(src + y1 * param.width * param.channel, c10, c11, c12,
                                                           param.width, cw, depX + param.cropOffsetX,
                                                           scaleX + param.cropOffsetX, mToTensorArgs.Mul());
            } else if (y0 - oy == 1) {
                std::swap(c00, c10);
                std::swap(c01, c11);
                std::swap(c02, c12);
                Compute3ChannelLine<InputType, OutputType>(src + y1 * param.width * param.channel, c10, c11, c12,
                                                           param.width, cw, depX + param.cropOffsetX,
                                                           scaleX + param.cropOffsetX, mToTensorArgs.Mul());
            }
            oy = y0;
            Add2LinesAndNorm<OutputType>(c00, c10, dst + RGB_CHANNEL_RED * cw * ch + (y - param.cropOffsetY) * cw, cw,
                                         sy0, sy1, (OutputType)mNormArgs.Mean()[RGB_CHANNEL_RED],
                                         (OutputType)mNormArgs.Scale()[RGB_CHANNEL_RED]);
            Add2LinesAndNorm<OutputType>(c01, c11, dst + RGB_CHANNEL_GREEN * cw * ch + (y - param.cropOffsetY) * cw, cw,
                                         sy0, sy1, (OutputType)mNormArgs.Mean()[RGB_CHANNEL_GREEN],
                                         (OutputType)mNormArgs.Scale()[RGB_CHANNEL_GREEN]);
            Add2LinesAndNorm<OutputType>(c02, c12, dst + RGB_CHANNEL_BLUE * cw * ch + (y - param.cropOffsetY) * cw, cw,
                                         sy0, sy1, (OutputType)mNormArgs.Mean()[RGB_CHANNEL_BLUE],
                                         (OutputType)mNormArgs.Scale()[RGB_CHANNEL_BLUE]);
        }
    }

    free(scaleX);
    free(scaleY);
    free(depX);
    free(depY);
    free(space);
}
ACCDATA_REGISTER_FUSION_OPERATOR(ToTensorResizeCropNormalize, ToTensorResizeCropNormalize);

}  // namespace accdata
}  // namespace acclib