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
#ifndef ACCDATA_SRC_CPP_OPERATOR_FUSION_TO_TENSOR_RESIZE_CROP_NORMALIZE_H_
#define ACCDATA_SRC_CPP_OPERATOR_FUSION_TO_TENSOR_RESIZE_CROP_NORMALIZE_H_

#include "common/utility.h"
#include "operator/image/crop_args.h"
#include "operator/image/resize_args.h"
#include "operator/image/to_tensor_args.h"
#include "operator/math/normalize_args.h"
#include "operator/operator.h"
#include "tensor/tensor_image.h"
#include "operator/operator_param_inner.h"

namespace acclib {
namespace accdata {

/**
 * @brief fusion operation that combine to_tensor/resize_crop/norm
 *
 * Argument:
 * - @see ref to ToTensorArgs, CropArgs, ResizeArgs and NormalizeArgs
 * SCHEMA END
 */
class ToTensorResizeCropNormalize : public Operator {
    using ResultType = float;  // Now assume the output datatype is float.
public:
    explicit ToTensorResizeCropNormalize(const OpSpec &spec) : Operator(spec)
    {
    }

    ~ToTensorResizeCropNormalize() = default;

    AccDataErrorCode Run(Workspace &ws) override;

private:
    AccDataErrorCode Setup(Workspace &ws);

    AccDataErrorCode GetOutputShape(const TensorList &input, TensorListShape &outputShape);

    template <typename InputType, typename OutputType>
    AccDataErrorCode ClassifyTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template <typename InputType, typename OutputType, TensorLayout InLayout, TensorLayout OutLayout>
    AccDataErrorCode AddTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    OperatorParam SetupParam();

    template <typename InputType, typename OutputType, TensorLayout InLayout, TensorLayout OutLayout>
    void RunTask(const InputType *input, OutputType *output, const OperatorParam &param, AccDataErrorCode &errCode);

    template <typename InputType, typename OutputType>
    void Kernel2NCHW(const InputType *input, OutputType *output, const OperatorParam &param);

    template <typename InputType, typename OutputType>
    inline int Compute3ChannelLine(const InputType *src, OutputType *dst0, OutputType *dst1, OutputType *dst2, int sw,
                                   int dw, int *dep, const OutputType *scale, const double div)
    {
        for (int i = 0; i < dw; i++) {
            int i0 = dep[i];
            int i1 = std::min(i0 + 1, sw - 1);
            auto sx0 = scale[i];
            // !!! calculate in `div * (sx0 * (src[i0*3+0] - src[i1*3+0]) + src[i1*3+0])` will lose precision
            dst0[i] = (sx0 * (OutputType)(src[i0 * RGB_CHANNELS + RGB_CHANNEL_RED] * div) +
                       (1 - sx0) * (OutputType)(src[i1 * RGB_CHANNELS + RGB_CHANNEL_RED] * div));
            dst1[i] = (sx0 * (OutputType)(src[i0 * RGB_CHANNELS + RGB_CHANNEL_GREEN] * div) +
                       (1 - sx0) * (OutputType)(src[i1 * RGB_CHANNELS + RGB_CHANNEL_GREEN] * div));
            dst2[i] = (sx0 * (OutputType)(src[i0 * RGB_CHANNELS + RGB_CHANNEL_BLUE] * div) +
                       (1 - sx0) * (OutputType)(src[i1 * RGB_CHANNELS + RGB_CHANNEL_BLUE] * div));
        }
        return 0;
    }

    template <typename T>
    inline int Add2LinesAndNorm(const T *s0, const T *s1, T *dst, int length, T scale0, T scale1, T mean, T scale)
    {
        for (int i = 0; i < length; i++) {
            dst[i] = scale * (scale0 * s0[i] + scale1 * s1[i] - mean);
        }
        return 0;
    }

private:
    image::Meta mInputMeta{};
    ToTensorArgs mToTensorArgs{};
    ResizeArgs mResizeArgs{};
    CropArgs mCropArgs{};
    NormalizeArgs mNormArgs{};
};

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_FUSION_TO_TENSOR_RESIZE_CROP_NORMALIZE_H_
