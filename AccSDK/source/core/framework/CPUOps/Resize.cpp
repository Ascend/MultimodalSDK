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
 * Description: Resize op on cpu.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <iostream>
#include <cmath>
#include <thread>
#include "acc/core/framework/CPUAccelerator.h"
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ThreadPool.h"
#include "acc/utils/ErrorCodeUtils.h"
using namespace Acc;

namespace {
constexpr size_t INDEX_ZERO = 0;
constexpr size_t INDEX_ONE = 1;
constexpr size_t INDEX_TWO = 2;
constexpr double BICUBICPARAM_A = -0.5;
constexpr int PRECISION_BITS = 22;
constexpr int INT_TWO = 2;
constexpr int INT_THREE = 3;
constexpr int SIZE_T_TWO = 2;
constexpr double DOUBLE_TWO = 2.0;
constexpr double DOUBLE_THREE = 3.0;
constexpr double DOUBLE_FOUR = 4.0;
constexpr double DOUBLE_FIVE = 5.0;
constexpr double DOUBLE_EIGHT = 8.0;
constexpr double DOUBLE_HALF = 0.5;
constexpr double DOUBLE_HALF_NEGATIVE = -0.5;
constexpr size_t RESIZE_DEFAULT_THREAD_NUMS = 16;
constexpr double DOUBLE_EPS = 1e-9;
constexpr double DOUBLE_SCALE = 1 << PRECISION_BITS;
uint8_t g_clampLookups[1280] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,   3,
    4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
    27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
    50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
    73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
    234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

uint8_t* g_clampLookupsHalf = &g_clampLookups[640];

// Limit the given value to the specified range
inline uint8_t ClampToUint8(int in)
{
    return g_clampLookupsHalf[in >> PRECISION_BITS];
}

// Calculate weights based on distance
inline double BicubicFilter(double x)
{
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((BICUBICPARAM_A + DOUBLE_TWO) * x - (BICUBICPARAM_A + DOUBLE_THREE)) * x * x + 1;
    }
    if (x < DOUBLE_TWO) {
        return (((x - DOUBLE_FIVE) * x + DOUBLE_EIGHT) * x - DOUBLE_FOUR) * BICUBICPARAM_A;
    }
    return 0.0;
}

// Calculate the boundary values and pixel weights needed for interpolation
void PreComputeCoefficient(size_t srcSize, size_t dstSize, size_t& kernelSize, std::vector<int>& bounds,
                           std::vector<double>& kernelCoefficient)
{
    double resizeRatio = static_cast<double>(srcSize) / static_cast<double>(dstSize);
    double filterScale = resizeRatio >= 1.0 ? resizeRatio : 1.0;
    // half of the kernelSize
    double radius = DOUBLE_TWO * filterScale;
    // total kernel size
    kernelSize = static_cast<size_t>(std::ceil(radius)) * SIZE_T_TWO + 1;
    // {xmin, xmax} for each output pixel，xmin：left bound of kernel，xmax：right bound of kernel
    bounds.resize(dstSize * SIZE_T_TWO);
    // each dst pixel needs kernelSize interpolation coefficients.
    kernelCoefficient.resize(dstSize * kernelSize, 0.0);
    for (size_t xx = 0; xx < dstSize; ++xx) {
        double center = (xx + DOUBLE_HALF) * resizeRatio;
        double inverseScale = 1.0 / filterScale;
        int xmin = static_cast<int>(center - radius + DOUBLE_HALF);
        if (xmin < 0) {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + radius + DOUBLE_HALF);
        if (xmax > static_cast<int>(srcSize)) {
            xmax = static_cast<int>(srcSize);
        }
        int validWidth = xmax - xmin;
        double* k = &kernelCoefficient[xx * kernelSize];
        double ww = 0.0;
        for (int x = 0; x < validWidth; ++x) {
            double weight = BicubicFilter((x + xmin - center + DOUBLE_HALF) * inverseScale);
            k[x] = weight;
            ww += weight;
        }
        if (std::abs(ww) > DOUBLE_EPS) {
            // normalize
            for (int x = 0; x < validWidth; ++x) {
                k[x] /= ww;
            }
        }
        // in case validWidth smaller than kernelSize
        for (int x = validWidth; x < static_cast<int>(kernelSize); ++x) {
            k[x] = 0.0;
        }
        bounds[xx * INT_TWO + 0] = xmin;
        bounds[xx * INT_TWO + 1] = validWidth;
    }
}

// cast double to long accelerate CPU compute
void NormalizeCoefficientVector(const std::vector<double>& kernelCoeIn, std::vector<long>& kernelCoeOut)
{
    kernelCoeOut.resize(kernelCoeIn.size());

    for (size_t x = 0; x < kernelCoeIn.size(); x++) {
        // Add or subtract 0.5 to round off
        if (kernelCoeIn[x] < 0.0) {
            kernelCoeOut[x] = static_cast<long>(DOUBLE_HALF_NEGATIVE + kernelCoeIn[x] * (DOUBLE_SCALE));
        } else {
            kernelCoeOut[x] = static_cast<long>(DOUBLE_HALF + kernelCoeIn[x] * (DOUBLE_SCALE));
        }
    }
}

void ComputeHorizontalSum(int widthBoundsEnd, const std::vector<long>& kernelCoeHorizNormalized, int& coeIndexHorizBase,
                          int& srcIndexBase, uint8_t* srcPtr, int& ss0, int& ss1, int& ss2)
{
    for (int x = 0; x < widthBoundsEnd; x++) {
        const long coeHoriz = kernelCoeHorizNormalized[coeIndexHorizBase];
        ss0 += srcPtr[srcIndexBase + INDEX_ZERO] * coeHoriz;
        ss1 += srcPtr[srcIndexBase + INDEX_ONE] * coeHoriz;
        ss2 += srcPtr[srcIndexBase + INDEX_TWO] * coeHoriz;
        srcIndexBase += INT_THREE;
        coeIndexHorizBase++;
    }
}

void Process(const std::vector<int>& boundsVert, const std::vector<int>& boundsHoriz, uint8_t* dstPtr, uint8_t* srcPtr,
             const std::vector<long>& kernelCoeHorizNormalized, const std::vector<long>& kernelCoeVertNormalized,
             size_t srcWidth, size_t dstWidth, int kernelSizeH, int kernelSizeW, int startRow, int endRow)
{
    // Iterate through each target point and calculate the pixel value
    const int initialBias = 1 << (PRECISION_BITS - 1);
    const int srcWidthStride = static_cast<int>(srcWidth) * INT_THREE;
    const int dstWidthStride = static_cast<int>(dstWidth) * INT_THREE;
    for (int yy = startRow; yy < endRow; yy++) {
        int heightBoundsStart = boundsVert[yy * INT_TWO + 0];
        int heightBoundsEnd = boundsVert[yy * INT_TWO + 1];
        for (int xx = 0; xx < static_cast<int>(dstWidth); xx++) {
            int widthBoundsStart = boundsHoriz[xx * INT_TWO + 0];
            int widthBoundsEnd = boundsHoriz[xx * INT_TWO + 1];
            int widthBoundsStride = widthBoundsStart * INT_THREE;
            int t0 = initialBias;
            int t1 = initialBias;
            int t2 = initialBias;
            int coeIndexVertBase = yy * kernelSizeH;
            for (int y = 0; y < heightBoundsEnd; y++) {
                int ss0 = initialBias;
                int ss1 = initialBias;
                int ss2 = initialBias;
                int srcIndexBase = ((y + heightBoundsStart) * srcWidthStride + widthBoundsStride);
                int coeIndexHorizBase = xx * kernelSizeW;
                ComputeHorizontalSum(widthBoundsEnd, kernelCoeHorizNormalized, coeIndexHorizBase, srcIndexBase, srcPtr,
                                     ss0, ss1, ss2);
                const long coeVert = kernelCoeVertNormalized[coeIndexVertBase + y];
                t0 += ClampToUint8(ss0) * coeVert;
                t1 += ClampToUint8(ss1) * coeVert;
                t2 += ClampToUint8(ss2) * coeVert;
            }
            int dstIndex = (yy * dstWidthStride + xx * INT_THREE);
            dstPtr[dstIndex + INDEX_ZERO] = ClampToUint8(t0);
            dstPtr[dstIndex + INDEX_ONE] = ClampToUint8(t1);
            dstPtr[dstIndex + INDEX_TWO] = ClampToUint8(t2);
        }
    }
};

void ResizeCalculate(const Tensor& src, Tensor& dst, int kernelSizeH, int kernelSizeW,
                     const std::vector<int>& boundsHoriz, const std::vector<double>& kernelCoefficientHoriz,
                     const std::vector<int>& boundsVert, const std::vector<double>& kernelCoefficientVert)
{
    auto srcShape = src.Shape();
    auto srcWidth = srcShape[INDEX_TWO];
    auto dstShape = dst.Shape();
    auto dstWidth = dstShape[INDEX_TWO];
    auto dstHeight = dstShape[INDEX_ONE];
    std::vector<long> kernelCoeHorizNormalized;
    NormalizeCoefficientVector(kernelCoefficientHoriz, kernelCoeHorizNormalized);
    std::vector<long> kernelCoeVertNormalized;
    NormalizeCoefficientVector(kernelCoefficientVert, kernelCoeVertNormalized);
    auto* dstPtr = static_cast<uint8_t*>(dst.Ptr());
    auto* srcPtr = static_cast<uint8_t*>(src.Ptr());
    auto threadNum = RESIZE_DEFAULT_THREAD_NUMS;
    std::vector<std::future<void>> futures;
    int rowsPerThread = static_cast<int>(dstHeight) / static_cast<int>(threadNum);
    int extraRows = static_cast<int>(dstHeight) % static_cast<int>(threadNum);
    auto& instance = ThreadPool::GetInstance();
    for (size_t t = 0; t < threadNum; ++t) {
        int startRow = static_cast<int>(t) * rowsPerThread;
        int endRow = (t == threadNum - 1) ? (startRow + rowsPerThread + extraRows) : (startRow + rowsPerThread);
        futures.push_back(instance.Submit(Process, boundsVert, boundsHoriz, dstPtr, srcPtr, kernelCoeHorizNormalized,
                                          kernelCoeVertNormalized, srcWidth, dstWidth, kernelSizeH, kernelSizeW,
                                          startRow, endRow));
    }
    instance.WaitAll(futures);
}
} // namespace

namespace Acc {
ErrorCode CPUAccelerator::Resize(ResizeContext& opCtx)
{
    std::vector<int> boundsHoriz;
    std::vector<double> kernelCoefficientHoriz;
    std::vector<int> boundsVert;
    std::vector<double> kernelCoefficientVert;
    size_t kernelSizeH = 0;
    size_t kernelSizeW = 0;
    const Tensor& src = opCtx.inputTensorRefs[0].get();
    Tensor& dst = opCtx.outputTensorRefs[0].get();
    auto srcShape = src.Shape();

    ErrorCode ret = SUCCESS;
    try {
        PreComputeCoefficient(srcShape[INDEX_ONE], opCtx.resizedH, kernelSizeH, boundsVert, kernelCoefficientVert);
        PreComputeCoefficient(srcShape[INDEX_TWO], opCtx.resizedW, kernelSizeW, boundsHoriz, kernelCoefficientHoriz);
        ResizeCalculate(src, dst, kernelSizeH, kernelSizeW, boundsHoriz, kernelCoefficientHoriz, boundsVert,
                        kernelCoefficientVert);
    } catch (const std::exception& e) {
        LogDebug << "There is a problem with the thread pool used in ResizeOnCpu."
                 << GetErrorInfo(ERR_INVALID_THREAD_POOL_STATUST);
        ret = ERR_INVALID_THREAD_POOL_STATUST;
    }

    return ret;
}
} // namespace Acc