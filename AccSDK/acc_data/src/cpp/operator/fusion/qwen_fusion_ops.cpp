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
 * @Date: 2025-7-10 14:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-7-10 14:00:00
 */

#include "qwen_fusion_ops.h"

#include "common/balance.h"
#include "common/tracer.h"
#include "operator/op_factory.h"

#include <climits>

namespace acclib {
namespace accdata {

AccDataErrorCode QwenFusionOp::Run(Workspace &ws)
{
    TRACE_BEGIN(qwen_fuison_op)
    AccDataErrorCode errCode = AccDataErrorCode::H_OK;
    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
                                   "Input out of range during fused operation execution.", errCode);
    if (input.IsEmpty() || !input.IsValid()) {
        ACCDATA_ERROR("Illegal input.");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }
    errCode = Setup(ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Failed to set up the to qwen2vl fusion operator.", errCode);

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
                                       "Failed to run Qwen2 Vl fusion task.", errCode);
    }
    errCode = pool.RunAll();
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to run Qwen2 VL op: ", errCode);

    TRACE_END(qwen_fuison_op)
    return errCode;
}

AccDataErrorCode QwenFusionOp::Setup(Workspace &ws)
{
    AccDataErrorCode errCode = AccDataErrorCode::H_OK;
    auto &spec = GetSpec();
    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Input out of range during fused operation execution.", errCode);
    if (input.NumTensors() < 1) {
        ACCDATA_ERROR("The number of input is empty.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    errCode = mInputMeta.Setup(input[0]);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Failed get mInputMeta for Qwen2 Vl fusion operation.", errCode);
    errCode = mNormalizeArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Failed get mNormalizeArgs for Qwen2 Vl fusion operation.", errCode);
    errCode = mQwenArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Failed get mQwenArgs for Qwen2 Vl fusion operation.", errCode);

    return errCode;
}

AccDataErrorCode QwenFusionOp::GetOutputShape(const TensorList &input, TensorListShape& outputShape)
{
    auto numSamples = mInputMeta.NumSamples();
    auto temporalPatchSize = mQwenArgs.TemporalPatchSize();
    auto patchSize = mQwenArgs.PatchSize();
    auto mergeSize = mQwenArgs.MergeSize();
    if (patchSize * mergeSize > mInputMeta.Height() ||
        patchSize * mergeSize > mInputMeta.Width()) {
        ACCDATA_ERROR("PatchSize * mergeSize should be less than input height and width.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }
    auto resizeInfo = SmartResize(patchSize * mergeSize);
    auto height = resizeInfo[0];
    auto width = resizeInfo[1];
    // if numSamples don't eq 1, precision is not right.
    if (height <= 0 || width <= 0 || numSamples != 1) {
        ACCDATA_ERROR("Resize height and width should be greater than 0, and numSamples should be 1.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }
    ACCDATA_DEBUG("resize height " << height << " resize width " << width);

    auto numChannels = mInputMeta.NumChannels();

    // if input has only 1 image, then need to duplicate the processed image twice
    if (numSamples == 1) {
        numSamples = 2; // 2 is Qwen tile
    }
    auto gridT = numSamples / temporalPatchSize;
    auto gridH = height / patchSize;
    auto gridW = width / patchSize;
    ACCDATA_DEBUG("Qwen grid_t = " << gridT << ", grid_h = " << gridH << ", grid_w = " << gridW);
    outputShape = TensorListShape(input.NumTensors(),
                                  {static_cast<size_t>(gridT * gridH * gridW),
                                   static_cast<size_t>(numChannels * temporalPatchSize * patchSize * patchSize)});
    return AccDataErrorCode::H_OK;
}


QwenFusionOp::Param QwenFusionOp::SetupParam()
{
    Param param;
    param.height = mInputMeta.Height();
    param.width = mInputMeta.Width();
    param.channel = mInputMeta.NumChannels();
    param.begin = 0;
    param.end = 0;

    auto resizeInfo = SmartResize(mQwenArgs.PatchSize() * mQwenArgs.MergeSize());
    param.resizeH = resizeInfo[0];
    param.resizeW = resizeInfo[1];

    return param;
}

std::vector<int64_t> QwenFusionOp::SmartResize(int64_t factor)
{
    auto height = mInputMeta.Height();
    auto width = mInputMeta.Width();
    auto resizeH = PyRound(static_cast<double>(height) / static_cast<double>(factor)) * factor;
    auto resizeW = PyRound(static_cast<double>(width) / static_cast<double>(factor)) * factor;
    if (resizeH * resizeW > mQwenArgs.MaxPixels()) {
        auto beta = std::sqrt((double)(height * width) / mQwenArgs.MaxPixels());
        resizeH = std::floor(height / beta / factor) * factor;
        resizeW = std::floor(width / beta / factor) * factor;
    } else if (resizeH * resizeW < mQwenArgs.MinPixels()) {
        auto beta = std::sqrt((double)mQwenArgs.MinPixels() / (height * width));
        resizeH = std::ceil(height * beta / factor) * factor;
        resizeW = std::ceil(width * beta / factor) * factor;
    }

    return { resizeH, resizeW };
}

template <typename InputType, typename OutputType>
AccDataErrorCode QwenFusionOp::ClassifyTask(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    auto inLayout = input.Layout();
    if (inLayout != TensorLayout::NHWC) {
        ACCDATA_ERROR("Unsupported input layout " << inLayout);
        return AccDataErrorCode::H_FUSIONOP_ERROR;
    }

    if (input.DataType() != TensorDataType::UINT8) {
        ACCDATA_ERROR("Unsupported input dataType " << input.DataType());
        return AccDataErrorCode::H_FUSIONOP_ERROR;
    }
    output.SetLayout(TensorLayout::PLAIN);

    return AddTask<InputType, OutputType, TensorLayout::NHWC>(pool, input, output);
}

template <typename InputType, typename OutputType, TensorLayout InLayout>
AccDataErrorCode QwenFusionOp::AddTask(ThreadPool &pool, const Tensor &input, Tensor &output)
{
    int numThreads = pool.NumThreads();
    auto numberSamples = mInputMeta.NumSamples();
    auto strideIn = NumElements(input.Shape(), 1);
    auto strideOut = NumElements(output.Shape(), 1);
    Param param = SetupParam();

    for (int64_t i = 0; i < numberSamples; ++i) {
        auto *in = input.RawDataPtr<InputType>() + i * strideIn;
        auto *out = output.RawDataPtr<OutputType>() + i * strideOut;
        for (int j = 0; j < numThreads; ++j) {
            /* split based on the height dimension to ensure parallelism. */
            Balance::Task range;
            auto errCode = Balance::Assign(param.resizeH, numThreads, j, range);
            ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.", errCode);
            if (range.begin >= range.end) {
                break;
            }
            param.begin = range.begin;
            param.end = range.end;
            /* Because the task is executed after it leaves this scope, so use value capture. */
            auto task = [this, in, out, param](int id, AccDataErrorCode &errCode) {
                RunTask<InputType, OutputType, InLayout>(in, out, param, errCode);
            };
            ACCDATA_DEBUG("Addtask sample: " << i << ", thread " << j);
            pool.AddTask(task);
        }
    }

    return AccDataErrorCode::H_OK;
}

template <typename InputType, typename OutputType, TensorLayout InLayout>
void QwenFusionOp::RunTask(const InputType *input, OutputType *output,
    const QwenFusionOp::Param &param, AccDataErrorCode &workerErr)
{
    if constexpr (InLayout == TensorLayout::NHWC) {
        KernelNHWC<InputType, OutputType>(input, output, param);
        workerErr = AccDataErrorCode::H_OK;
    } else {
        ACCDATA_ERROR("Unsupported result layout " << InLayout);
        workerErr = AccDataErrorCode::H_FUSIONOP_ERROR;
    }
    return;
}

int QwenFusionOp::PrecomputeCoeffs(int inSize, int outSize, int **boundsPtr, double **coeffsPtr)
{
    double filterScale; // scale to find surround pixel in origin image
    double scale; // scale to find center pixel in origin image
    filterScale = scale = static_cast<double>(inSize) / outSize;
    if (ACCDATA_UNLIKELY(filterScale < 1.0)) {
        filterScale = 1.0;
    }

    /* the length of resampling filter is 2.0 */
    auto support = 2.0 * filterScale;
    /* maximum number of coeffs for each pixel is 2 */
    int coeffSize = static_cast<int>(ceil(support)) * 2 + 1;
    if (ACCDATA_UNLIKELY(outSize > INT_MAX / (coeffSize * static_cast<int>(sizeof(double))))) {
        return 0;
    }

    auto coeffs = (double *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(outSize * coeffSize, sizeof(double)));
    auto bounds = (int *)aligned_alloc(ACCDATA_ALIGN_SIZE, AlignUp(outSize * BOUND_SIZE, sizeof(int)));

    auto ss = 1.0 / filterScale;
    for (auto i = 0; i < outSize; i++) {
        /* center before resize */
        auto center = (i + POINT_FIVE) * scale;
        /* calculate lower bound and filter window size for each pixel */
        auto lower = std::max((int)(center - support + POINT_FIVE), 0);
        auto delta = std::min((int)(center + support + POINT_FIVE), inSize) - lower;
        bounds[i * BOUND_SIZE + 0] = lower;
        bounds[i * BOUND_SIZE + 1] = delta;

        /* calculate coefficients for each pixel */
        auto coeff = &coeffs[i * coeffSize];
        auto ww = 0.0;
        int j;

        for (j = 0; j < delta; j++) {
            auto w = BicubicFilter((lower + j - center + POINT_FIVE) * ss);
            coeff[j] = w;
            ww += w;
        }
        /* normalize coefficients for each pixel */
        if (ww != 0.0) {
            for (j = 0; j < delta; j++) {
                coeff[j] /= ww;
            }
        }
        for (; j < coeffSize; j++) {
            coeff[j] = 0;
        }
    }

    *boundsPtr = bounds;
    *coeffsPtr = coeffs;
    return coeffSize;
}

/**
 * @brief Resize the input image in horizontal direction.
 *
 * @tparam InputType Type of the input data (e.g., uint8_t).
 * @tparam OutputType Type of the output data (e.g., float).
 * @param input Pointer to the input data.
 * @param output Pointer to the output data.
 * @param param Qwen Parameters
 * @param resizeCoeffs Precomputed coefficients for resizing.
*/
template <typename InputType, typename OutputType>
void QwenFusionOp::KernelNHWCHorizontal(const InputType *input, OutputType *output, const QwenFusionOp::Param &param,
    resizeKernelCoeffs& resizeCoeffs)
{
    double *coeffsX = resizeCoeffs.coeffsX;
    int *boundsX = resizeCoeffs.boundsX;
    // First used row in the source image
    auto yStart = resizeCoeffs.boundsY[param.begin * BOUND_SIZE];
    // Last used row in the source image
    auto yEnd = resizeCoeffs.boundsY[param.end * BOUND_SIZE - BOUND_SIZE] +
        resizeCoeffs.boundsY[param.end * BOUND_SIZE - 1];
    auto coeffSizeX = resizeCoeffs.coeffSizeX;
    auto intCoeffsX = reinterpret_cast<long *>(coeffsX);

    for (auto yy = yStart; yy < yEnd; ++yy) {
        for (int xx = 0; xx < param.resizeW; ++xx) {
            int xmin = boundsX[xx * BOUND_SIZE + 0];
            int xmax = boundsX[xx * BOUND_SIZE + 1];
            auto wu = &intCoeffsX[xx * coeffSizeX];
            int ss0 = 1 << (PRECISION_BITS - 1);
            int ss1 = ss0;
            int ss2 = ss0;
            for (int x = 0; x < xmax; x++) {
                int offset = yy * param.width * RGB_CHANNELS + (xmin + x) * RGB_CHANNELS;
                ss0 += static_cast<uint8_t>(input[offset + RGB_CHANNEL_RED]) * wu[x];
                ss1 += static_cast<uint8_t>(input[offset + RGB_CHANNEL_GREEN]) * wu[x];
                ss2 += static_cast<uint8_t>(input[offset + RGB_CHANNEL_BLUE]) * wu[x];
            }
            int offset = (yy - yStart) * param.resizeW * RGB_CHANNELS + xx * RGB_CHANNELS;
            output[offset + RGB_CHANNEL_RED] = Clip8(ss0);
            output[offset + RGB_CHANNEL_GREEN] = Clip8(ss1);
            output[offset + RGB_CHANNEL_BLUE] = Clip8(ss2);
        }
    }
}

/**
 * @brief Resize the input image in vertical direction, then do totensor normalization and transpose.
 * The output is in CHW format.
 *
 * @tparam InputType Type of the input data (e.g., uint8_t).
 * @tparam OutputType Type of the output data (e.g., float).
 * @param input Pointer to the input data.
 * @param output Pointer to the output data.
 * @param param Parameters for resizing and transposing.
 * @param resizeCoeffs Precomputed coefficients for resizing.
 */
template <typename InputType, typename OutputType>
void QwenFusionOp::KernelNHWCVertical(const InputType *input, OutputType *output, const QwenFusionOp::Param &param,
    resizeKernelCoeffs& resizeCoeffs)
{
    auto resized_row = (OutputType *)aligned_alloc(ACCDATA_ALIGN_SIZE,
        AlignUp(param.resizeW * RGB_CHANNELS, sizeof(OutputType)));
    int64_t hdim[RGB_CHANNELS] = {mQwenArgs.PatchSize(), mQwenArgs.MergeSize(),
        param.resizeH / (mQwenArgs.MergeSize() * mQwenArgs.PatchSize())};
    int64_t wdim[RGB_CHANNELS] = {mQwenArgs.PatchSize(), mQwenArgs.MergeSize(),
        param.resizeW / (mQwenArgs.MergeSize() * mQwenArgs.PatchSize())};
    int64_t tps = mQwenArgs.TemporalPatchSize();
    Tranpose quickTranspose(hdim, wdim, tps);
    int *boundsY = resizeCoeffs.boundsY;
    auto intCoeffsY = reinterpret_cast<long *>(resizeCoeffs.coeffsY);
    auto coeffSizeY = resizeCoeffs.coeffSizeY;
    auto yStart = resizeCoeffs.boundsY[param.begin * BOUND_SIZE];

    for (auto yy = param.begin; yy < param.end; ++yy) {
        auto wv = &intCoeffsY[yy * coeffSizeY];
        int ymin = boundsY[yy * BOUND_SIZE + 0];
        int ymax = boundsY[yy * BOUND_SIZE + 1];
        for (int xx = 0; xx < param.resizeW; ++xx) {
            int t0 = 1 << (PRECISION_BITS - 1);
            int t1 = t0;
            int t2 = t0;

            for (int y = 0; y < ymax; y++) {
                int offset = (y + ymin - yStart) * param.resizeW * RGB_CHANNELS + xx * RGB_CHANNELS;
                t0 += (static_cast<uint8_t>(input[offset + RGB_CHANNEL_RED])) * wv[y];
                t1 += (static_cast<uint8_t>(input[offset + RGB_CHANNEL_GREEN])) * wv[y];
                t2 += (static_cast<uint8_t>(input[offset + RGB_CHANNEL_BLUE])) * wv[y];
            }

            // hwc->chw && bgr->rgb && rescale
            double mul = ToTensorArgs::NORM_FACTOR;
            auto mean = mNormalizeArgs.Mean();
            auto scale = mNormalizeArgs.Scale();
            resized_row[RGB_CHANNEL_RED * param.resizeW + xx] =
                    ((OutputType)(Clip8(t0) * mul) - mean[RGB_CHANNEL_RED]) * scale[RGB_CHANNEL_RED];
            resized_row[RGB_CHANNEL_GREEN * param.resizeW + xx] =
                    ((OutputType)(Clip8(t1) * mul) - mean[RGB_CHANNEL_GREEN]) * scale[RGB_CHANNEL_GREEN];
            resized_row[RGB_CHANNEL_BLUE * param.resizeW + xx] =
                    ((OutputType)(Clip8(t2) * mul) - mean[RGB_CHANNEL_BLUE]) * scale[RGB_CHANNEL_BLUE];
        }

        quickTranspose.Apply(resized_row + RGB_CHANNEL_RED * param.resizeW,
            output, RGB_CHANNEL_RED, yy, hdim, wdim, tps);
        quickTranspose.Apply(resized_row + RGB_CHANNEL_GREEN * param.resizeW,
            output, RGB_CHANNEL_GREEN, yy, hdim, wdim, tps);
        quickTranspose.Apply(resized_row + RGB_CHANNEL_BLUE * param.resizeW,
            output, RGB_CHANNEL_BLUE, yy, hdim, wdim, tps);
    }

    free(resized_row);
}

template <typename InputType, typename OutputType>
void QwenFusionOp::KernelNHWC(const InputType *input, OutputType *output, const QwenFusionOp::Param &param)
{
    ACCDATA_DEBUG("Running KernelNHWC2Pass");
    TRACE_BEGIN(FusionComputeOpt)
    resizeKernelCoeffs resizeCoeffs;

    resizeCoeffs.coeffSizeX = PrecomputeCoeffs(param.width, param.resizeW,
        &resizeCoeffs.boundsX, &resizeCoeffs.coeffsX);
    NormalizeCoeffs(param.resizeW, resizeCoeffs.coeffSizeX, resizeCoeffs.coeffsX);
    resizeCoeffs.coeffSizeY = PrecomputeCoeffs(param.height, param.resizeH,
        &resizeCoeffs.boundsY, &resizeCoeffs.coeffsY);
    NormalizeCoeffs(param.resizeH, resizeCoeffs.coeffSizeY, resizeCoeffs.coeffsY);
    // First used row in the source image
    auto yStart = resizeCoeffs.boundsY[param.begin * BOUND_SIZE];
    // Last used row in the source image
    auto yEnd = resizeCoeffs.boundsY[param.end * BOUND_SIZE - BOUND_SIZE] +
        resizeCoeffs.boundsY[param.end * BOUND_SIZE - 1];
    auto tmp_output = (InputType *)aligned_alloc(ACCDATA_ALIGN_SIZE,
        AlignUp((yEnd - yStart) * param.resizeW * RGB_CHANNELS, sizeof(InputType)));

    KernelNHWCHorizontal(input, tmp_output, param, resizeCoeffs);
    KernelNHWCVertical(tmp_output, output, param, resizeCoeffs);

    free(resizeCoeffs.boundsX);
    free(resizeCoeffs.boundsY);
    free(resizeCoeffs.coeffsX);
    free(resizeCoeffs.coeffsY);
    free(tmp_output);
    
    TRACE_END(FusionComputeOpt)
}
ACCDATA_REGISTER_FUSION_OPERATOR(QwenFusionOp, QwenFusionOp);

}  // namespace accdata
}  // namespace acclib