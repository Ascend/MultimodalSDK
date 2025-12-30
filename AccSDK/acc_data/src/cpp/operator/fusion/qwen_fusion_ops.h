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
#ifndef ACCDATA_OPERATOR_IMAGE_QWEN_FUSION_OPS_H
#define ACCDATA_OPERATOR_IMAGE_QWEN_FUSION_OPS_H
#include <cstdint>
#include "operator/operator.h"
#include "operator/image/resize_args.h"
#include "operator/image/crop_args.h"
#include "operator/image/to_tensor_args.h"

#include "operator/math/normalize_args.h"
#include "tensor/tensor_image.h"
#include "qwen_args.h"

namespace acclib {
namespace accdata {

constexpr double POINT_FIVE = 0.5; // used in BicubicFilter or other case.
constexpr int BOUND_SIZE = 2; // maximum number of coeffs for each pixel is 2

/**
 * @brief fusion operation for qwen2-vl image preprocess that combine
 * to_numpy_array/resize/rescale/normalize/to_chw/tile
 *
 * Argument:
 * - @see ref to
 * SCHEMA END
 */
class QwenFusionOp : public Operator {
    using ResultType = float; // Now assume the output datatype is float.
public:
    explicit QwenFusionOp(const OpSpec &spec) : Operator(spec) {}

    ~QwenFusionOp() = default;

    AccDataErrorCode Run(Workspace &ws) override;

private:
    struct Param {
        int64_t height = 0;
        int64_t width = 0;
        int64_t resizeH = 0;
        int64_t resizeW = 0;
        int64_t channel = 0;
        /* Task range */
        int64_t begin = 0;
        int64_t end = 0;
    };

    struct resizeKernelCoeffs {
        double *coeffsX;
        int *boundsX;
        int coeffSizeX;
        double *coeffsY;
        int *boundsY;
        int coeffSizeY;
    };

    AccDataErrorCode Setup(Workspace &ws);

    AccDataErrorCode GetOutputShape(const TensorList &input, TensorListShape& outputShape);

    Param SetupParam();

    std::vector<int64_t> SmartResize(int64_t factor = 2 * 14);

    template <typename InputType, typename OutputType>
    AccDataErrorCode ClassifyTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template <typename InputType, typename OutputType, TensorLayout InLayout>
    AccDataErrorCode AddTask(ThreadPool &pool, const Tensor &input, Tensor &output);

    template <typename InputType, typename OutputType, TensorLayout InLayout>
    void RunTask(const InputType *input, OutputType *output, const Param &param, AccDataErrorCode &workerErr);

    template <typename InputType, typename OutputType>
    void KernelNHWCHorizontal(const InputType *input, OutputType *output, const Param &param,
        resizeKernelCoeffs& resizeCoeffs);

    template <typename InputType, typename OutputType>
    void KernelNHWCVertical(const InputType *input, OutputType *output, const Param &param,
        resizeKernelCoeffs& resizeCoeffs);

    template <typename InputType, typename OutputType>
    void KernelNHWC(const InputType *input, OutputType *output, const Param &param);

    static inline double BicubicFilter(double x)
    {
        static constexpr double a = -POINT_FIVE;
        if (x < 0.0) {
            x = -x;
        }
        if (x < 1.0) {
            return ((a + 2.0F) * x - (a + 3.0F)) * x * x + 1.0;
        }
        if (x < 2.0F) {
            return (((x - 5.0F) * x + 8.0F) * x - 4.0F) * a;
        }
        return 0.0;
    }

    const int PRECISION_BITS = 32 - 8 - 2;
    inline uint8_t Clip8(int in)
    {
        uint8_t *clip8Lookups = &mClip8Lookups[640];
        return clip8Lookups[in >> PRECISION_BITS];
    }

     /**
      * precompute coefficient w(x) and bound for each dimension
      * @param inSize input size
      * @param outSize output size
      * @param boundsPtr bounds for dimension x, e.g. x_min as lower bound of the filter and x_max as the size of filter
      * @param coeffsPtr coefficient for dimension x
      * @return maximum number of coeffs for current pixel
      */
    int PrecomputeCoeffs(int inSize, int outSize, int **boundsPtr, double **coeffsPtr);

    void NormalizeCoeffs(int outSize, int ksize, double *coeffs)
    {
        auto longCoeffs = reinterpret_cast<long *>(coeffs);
        for (auto x = 0; x < outSize * ksize; x++) {
            if (coeffs[x] < 0) {
                longCoeffs[x] = static_cast<int>(-POINT_FIVE + coeffs[x] * (1 << PRECISION_BITS));
            } else {
                longCoeffs[x] = static_cast<int>(POINT_FIVE + coeffs[x] * (1 << PRECISION_BITS));
            }
        }
    }

private:
    image::Meta mInputMeta;
    NormalizeArgs mNormalizeArgs;
    QwenArgs mQwenArgs;
    /* Handles values form -640 to 639. */
    uint8_t mClip8Lookups[1280] = {
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,   3,   4,   5,
        6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
        23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
        57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
        74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
        142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
        159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
        227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
        244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255,
    };
};

class Tranpose {
public:
    Tranpose(int64_t hdim[3], int64_t wdim[3], int64_t b)
    {
        height = hdim[0] * hdim[1] * hdim[2ULL];
        width = wdim[0] * wdim[1] * wdim[2ULL];
        hstride0 = wdim[0];
        hstride1 = wdim[0] * wdim[1] * hdim[0] * b * 3LL;
        hstride2 = hdim[0] * hdim[1] * width * b * 3LL;
        bstride = hdim[0] * wdim[0];
        cstride = bstride * b;
        wstride0 = 1;
        wstride1 = cstride * 3LL;
        wstride2 = wstride1 * wdim[1] * hdim[1];
    }

    template <typename T> void ApplyBatch(const T *src, T *dst, int64_t wdim[3])
    {
        for (int w2 = 0; w2 < wdim[2ULL]; w2++) {
            T *dp2 = dst + w2 * wstride2;
            for (int w1 = 0; w1 < wdim[1]; w1++) {
                T *dp1 = dp2 + w1 * wstride1;
                for (int w0 = 0; w0 < wdim[0]; w0++) {
                    dp1[w0] = *src++;
                }
            }
        }
    }

    template <typename T> void Apply(const T *src, T *dst, int c, int h, int64_t hdim[3], int64_t wdim[3], int64_t b)
    {
        int h0 = h % hdim[0];
        int h1 = (h / hdim[0]) % hdim[1];
        int h2 = h / (hdim[0] * hdim[1]);
        for (int j = 0; j < b; j++) {
            T *dp = dst + h0 * hstride0 + h1 * hstride1 + h2 * hstride2 + c * cstride + j * bstride;
            const T *sp = src;
            ApplyBatch(sp, dp, wdim);
        }
    }

private:
    int64_t height = 0;   // = hdim[0] * hdim[1] * hdim[2];
    int64_t width = 0;    // = wdim[0] * wdim[1] * wdim[2];
    int64_t hstride0 = 0; // = wdim[0];
    int64_t hstride1 = 0; // = wdim[0] * wdim[1] * hdim[0] * b * 3;
    int64_t hstride2 = 0; // = hdim[0] * hdim[1] * width * b * 3;
    int64_t bstride = 0;  // = hdim[0] * wdim[0];
    int64_t cstride = 0;  // = bstride * b;
    int64_t wstride0 = 1; // = 1;
    int64_t wstride1 = 0; // = cstride * 3;
    int64_t wstride2 = 0; // = wstride1 * wdim[1] * hdim[1];
};

}  // namespace accdata
}  // namespace acclib


#endif // ACCDATA_OPERATOR_IMAGE_QWEN_FUSION_OPS_H
