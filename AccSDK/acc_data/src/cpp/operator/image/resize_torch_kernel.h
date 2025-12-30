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
 * @Date: 2025-2-17 16:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-17 16:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_TORCH_KERNEL_H_
#define ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_TORCH_KERNEL_H_

#include <array>

#include "operator/operator.h"
#include "pipeline/workspace/workspace.h"
#include "tensor/tensor_image.h"
#include "common/balance.h"

namespace acclib {
namespace accdata {

constexpr double CENTER_ALIGN_PARAM = 0.5;
constexpr int ROW_CNT = 2;  // bilinear计算,两个临时数组,分别存放源坐标y1、y2两行上的像素值
constexpr int CALC_2_ROW_GAP = 2;  // 此轮与上一轮目标行映射到源图中的四行无重叠,上下数组需要重新计算
constexpr int CALC_1_ROW_GAP = 1;  // 此轮与上一轮目标行映射到源图存在重叠,上一轮的下像素值可用作此轮的上像素值
constexpr int LEFT_UPPER = 0;     // bicubic插值方式坐标
constexpr int CURR_PIX = 1;
constexpr int RIGHT_LOWER = 2;
constexpr int RIGHT_LOWER2 = 3;

template <typename T>
struct CalcPixParams {
    int sw;
    int sh;
    int tw;
    int th;
    int cropOffsetY;
    T heightScale;
    int channels;
    Balance::Task range;
};

template <typename T>
void CalcPixBilinear(CalcPixParams<T> param, std::array<std::vector<T>, 2ULL> lambdas,
                     std::array<std::vector<int>, 2ULL> iws, const T* srcPtr, T* dstPtr, AccDataErrorCode &errCode)
{
    T* space4UpperLowerValue = reinterpret_cast<T*>(aligned_alloc(256, param.tw * ROW_CNT * sizeof(T)));
    if (space4UpperLowerValue == nullptr) {
        ACCDATA_ERROR("In torch bilinar alloc space failed.");
        errCode = AccDataErrorCode::H_SINGLEOP_ERROR;
        return;
    }

    T* c0 = space4UpperLowerValue;  // 存放上方像素值的数组,数组长度为tw
    T* c1 = space4UpperLowerValue + param.tw;  // 存放下方像素值

    for (uint64_t idx = param.range.begin; idx < static_cast<uint64_t>(param.range.end); idx++) {
        int s = idx / param.channels;  // 当前batch编号
        int c = idx % param.channels;  // 当前channel编号

        auto spPtr = srcPtr + (s * param.channels + c) * param.sh * param.sw;  // srcTensor中起始位置
        auto dpPtr = dstPtr + (s * param.channels + c) * param.th * param.tw;  // dstTensor中起始位置

        int oi = -2;  // 第一轮上下像素值数组都需要计算，确保不重叠
        // 计算目标像素对应源图像的垂直方向像素点位置
        for (uint64_t oh = param.cropOffsetY; oh < static_cast<uint64_t>(param.cropOffsetY + param.th); ++oh) {
            T fh = std::max((oh + CENTER_ALIGN_PARAM) * param.heightScale - CENTER_ALIGN_PARAM, 0.0);
            int ih0 = static_cast<int>(fh);  // 上方像素坐标,取计算出的点的整数部分
            int ih1 = std::min(ih0 + 1, param.sh - 1);  // 下方像素坐标,防越界
            T h0lambda = 1.0 - (fh - ih0);  // 插值权重,线性计算x点像素值 = 左侧值 * h0lambda + 右侧值 * h1lambda
            T h1lambda = 1.0 - h0lambda;
            // 计算上下插值数组
            if (ih0 - oi >= CALC_2_ROW_GAP) {
                for (uint64_t ow = 0; ow < static_cast<uint64_t>(param.tw); ++ow) {
                    c0[ow] = lambdas[0][ow] * spPtr[ih0 * param.sw + iws[0][ow]] +
                             lambdas[1][ow] * spPtr[ih0 * param.sw + iws[1][ow]];
                    c1[ow] = lambdas[0][ow] * spPtr[ih1 * param.sw + iws[0][ow]] +
                             lambdas[1][ow] * spPtr[ih1 * param.sw + iws[1][ow]];
                }
            } else if (ih0 - oi == CALC_1_ROW_GAP) {
                std::swap(c0, c1);  // 上一轮与此轮重叠,上一轮的下像素值可以直接用作此轮的上像素值
                for (uint64_t ow = 0; ow < static_cast<uint64_t>(param.tw); ++ow) {
                    c1[ow] = lambdas[0][ow] * spPtr[ih1 * param.sw + iws[0][ow]] +
                             lambdas[1][ow] * spPtr[ih1 * param.sw + iws[1][ow]];
                }
            }
            oi = ih0;
            // 根据上下像素值,线性计算本坐标的像素值
            for (uint64_t ow = 0; ow < static_cast<uint64_t>(param.tw); ++ow) {
                dpPtr[(oh - param.cropOffsetY) * param.tw + ow] = h0lambda * c0[ow] + h1lambda * c1[ow];
            }
        }
    }

    free(space4UpperLowerValue);
    errCode = AccDataErrorCode::H_OK;
}

/**
 * resize crop in bilinear mode similar to pytorch
 *
 * @param src       source tensor to tackle
 * @param dst       dest tensor
 * @param rh        resize height
 * @param rw        resize width
 * @param th        target height of the output tensor
 * @param tw        target width of the output tensor
 * @param nts       number of threads
 * @return
 */
template <typename T>
AccDataErrorCode TorchBilinear(const Tensor& src, Tensor& dst, int rh, int rw, int th, int tw, image::Meta& inputMeta,
                               Workspace& ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    // 中心裁剪，计算从图像左边缘到裁剪区域左边缘的距离，/2以左右对称
    int cropOffsetX = (rw - tw) / 2;
    int cropOffsetY = (rh - th) / 2;
    // NCHW
    int batchsize = static_cast<int>(inputMeta.NumSamples());
    int channels = static_cast<int>(inputMeta.NumChannels());
    int sh = static_cast<int>(inputMeta.Height());
    int sw = static_cast<int>(inputMeta.Width());

    std::vector<int> iw0(tw);  // 左侧像素索引
    std::vector<int> iw1(tw);  // 右侧像素索引
    std::vector<T> w0lambda(tw);  // 与左侧邻域像素插值权重
    std::vector<T> w1lambda(tw);  // 与右侧邻域像素插值权重

    T widthScale = static_cast<T>(sw) / static_cast<T>(rw);  // 水平方向缩放系数
    T heightScale = static_cast<T>(sh) / static_cast<T>(rh);  // 垂直方向缩放系数

    // ow设置为uint64_t，以避免溢出问题，下面的for循环也是一样的
    for (uint64_t ow = 0; ow < static_cast<uint64_t>(tw); ++ow) {
        int iw = cropOffsetX + ow;
        // 计算目标像素对应源图像的水平方向像素点位置
        T fw = std::max((iw + CENTER_ALIGN_PARAM) * widthScale - CENTER_ALIGN_PARAM, 0.0);
        iw0[ow] = static_cast<int>(fw);  // 左侧像素坐标，取计算出的点的整数部分
        iw1[ow] = std::min(iw0[ow] + 1, sw - 1);  // 右侧像素坐标，防止越界
        w0lambda[ow] = 1.0 - (fw - iw0[ow]);  // 计算与左侧邻域像素插值权重
        w1lambda[ow] = 1.0 - w0lambda[ow];  // 计算与右侧邻域像素插值权重
    }

    auto srcPtr = src.RawDataPtr<T>();
    auto dstPtr = dst.RawDataPtr<T>();

    int nts = 0;
    Balance::Task range = {0, 0};
    errCode = ws.GetNumThreads(nts);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get threads number.", errCode);
    for (int t = 0; t < nts; t++) {
        errCode = Balance::Assign(batchsize * channels, nts, t, range);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.", errCode);
        if (range.begin >= range.end) {
            break;
        }

        auto task = [sw, sh, tw, th, heightScale, channels, cropOffsetY, iw0, iw1, w0lambda, w1lambda, srcPtr, dstPtr,
                range](int id, AccDataErrorCode &errCode) {
            CalcPixBilinear({sw, sh, tw, th, cropOffsetY, heightScale, channels, range}, {w0lambda, w1lambda},
                            {iw0, iw1}, srcPtr, dstPtr, errCode);
        };

        ws.GetThreadPool().AddTask(task);
    }
    errCode = ws.GetThreadPool().RunAll();
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to run task", errCode);

    return AccDataErrorCode::H_OK;
}

inline int Clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

/*
 * 双三次卷积权重公式: w(x) = (a+2)|x|^3-(a+3)|x|^2+1    |x|<=1,
 *                         a|x|^3-5a|x|^2+8a|x|-4a    1<|x|<2,
 *                         0                          otherwise
 * 取映射点周围最近的16个像素,映射点坐标(i,j),向下取整得(i',j'),横坐标范围[i'-1,j'+2],纵坐标范围[i'-1,j'+2]
 * 入参x:当处理水平方向时x为i-i', 纵向时x为j-j',x>0
 * 入参coeffs:长度为4的数组，存储当前行/列的权重
 */
template <typename T>
inline void InterpolateCubic(T x, T* coeffs)
{
    const float a = -0.75f;  // 平滑参数
    // i'-1列和j'-1行像素到映射点距离为x+1,符合公式a|x’|^3-5a|x‘|^2+8a|x’|-4a,此处x'=x+1,下方同理
    coeffs[0] = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a;
    coeffs[1] = ((a + 2) * x - (a + 3)) * x * x + 1;
    coeffs[2] = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];  // 保证系数和为1,符合插值数学特性
}

template <typename T>
struct T4 {
    std::array<T, 4ULL> values;
};

template <typename T>
void CalcPixBicubic(CalcPixParams<T> param, std::array<std::vector<int>, 4> iws, std::vector<T4<T>> scaleX,
                    const T* srcPtr, T* dstPtr, AccDataErrorCode &errCode = AccDataErrorCode::H_OK)
{
    for (int idx = param.range.begin; idx < param.range.end; idx++) {
        int s = idx / param.channels;
        int c = idx % param.channels;

        auto spPtr = srcPtr + (s * param.channels + c) * param.sh * param.sw;
        auto dpPtr = dstPtr + (s * param.channels + c) * param.th * param.tw;

        for (int oh = param.cropOffsetY; oh < param.cropOffsetY + param.th; ++oh) {
            T fh = (oh + CENTER_ALIGN_PARAM) * param.heightScale - CENTER_ALIGN_PARAM;
            int fh1 = std::min(static_cast<int>(floorf(fh)), param.sh - 1);  //  向下取整并防止越界
            T lambda = std::min(fh - fh1, static_cast<T>(1));
            T4<T> scaleY;

            InterpolateCubic(lambda, scaleY.values.data());

            int ih0 = Clip(fh1 - 1, 0, param.sh);  // 纵坐标分别取上1，下1，下2和fh1当前所在坐标,防止越界
            int ih1 = Clip(fh1, 0, param.sh);
            int ih2 = Clip(fh1 + 1, 0, param.sh);
            int ih3 = Clip(fh1 + 2, 0, param.sh);

            std::vector<T> c0(param.tw);
            std::vector<T> c1(param.tw);
            std::vector<T> c2(param.tw);
            std::vector<T> c3(param.tw);

            // 假设取映射点P周围16个像素,a00到a33(P位于a11),value(P)=∑(0<=i<=3)∑(0<=j<=3)aij * W(i) * W(j)
            for (int ow = 0; ow < param.tw; ++ow) {
                c0[ow] = scaleX[ow].values[LEFT_UPPER] * spPtr[ih0 * param.sw + iws[LEFT_UPPER][ow]] +
                         scaleX[ow].values[CURR_PIX] * spPtr[ih0 * param.sw + iws[CURR_PIX][ow]] +
                         scaleX[ow].values[RIGHT_LOWER] * spPtr[ih0 * param.sw + iws[RIGHT_LOWER][ow]] +
                         scaleX[ow].values[RIGHT_LOWER2] * spPtr[ih0 * param.sw + iws[RIGHT_LOWER2][ow]];

                c1[ow] = scaleX[ow].values[LEFT_UPPER] * spPtr[ih1 * param.sw + iws[LEFT_UPPER][ow]] +
                         scaleX[ow].values[CURR_PIX] * spPtr[ih1 * param.sw + iws[CURR_PIX][ow]] +
                         scaleX[ow].values[RIGHT_LOWER] * spPtr[ih1 * param.sw + iws[RIGHT_LOWER][ow]] +
                         scaleX[ow].values[RIGHT_LOWER2] * spPtr[ih1 * param.sw + iws[RIGHT_LOWER2][ow]];

                c2[ow] = scaleX[ow].values[LEFT_UPPER] * spPtr[ih2 * param.sw + iws[LEFT_UPPER][ow]] +
                         scaleX[ow].values[CURR_PIX] * spPtr[ih2 * param.sw + iws[CURR_PIX][ow]] +
                         scaleX[ow].values[RIGHT_LOWER] * spPtr[ih2 * param.sw + iws[RIGHT_LOWER][ow]] +
                         scaleX[ow].values[RIGHT_LOWER2] * spPtr[ih2 * param.sw + iws[RIGHT_LOWER2][ow]];

                c3[ow] = scaleX[ow].values[LEFT_UPPER] * spPtr[ih3 * param.sw + iws[LEFT_UPPER][ow]] +
                         scaleX[ow].values[CURR_PIX] * spPtr[ih3 * param.sw + iws[CURR_PIX][ow]] +
                         scaleX[ow].values[RIGHT_LOWER] * spPtr[ih3 * param.sw + iws[RIGHT_LOWER][ow]] +
                         scaleX[ow].values[RIGHT_LOWER2] * spPtr[ih3 * param.sw + iws[RIGHT_LOWER2][ow]];
            }

            for (int ow = 0; ow < param.tw; ++ow) {
                dpPtr[(oh - param.cropOffsetY) * param.tw + ow] = scaleY.values[LEFT_UPPER] * c0[ow] +
                      scaleY.values[CURR_PIX] * c1[ow] + scaleY.values[RIGHT_LOWER] * c2[ow] +
                      scaleY.values[RIGHT_LOWER2] * c3[ow];
            }
        }
    }
}

/**
 * resize crop in bicubic mode similar to pytorch
 * @param src       source tensor to tackle
 * @param dst       dest tensor
 * @param rh        resize height
 * @param rw        resize width
 * @param th        target height of the output tensor
 * @param tw        target width of the output tensor
 * @param nts       number of threads
 * @return
 */
template <typename T>
AccDataErrorCode TorchBicubic(const Tensor& src, Tensor& dst, int rh, int rw, int th, int tw, image::Meta& inputMeta,
                              Workspace& ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    int cropOffsetX = (rw - tw) / 2;  // 中心裁剪，计算从图像左边缘到裁剪区域左边缘的距离，/2以左右对称
    int cropOffsetY = (rh - th) / 2;

    int batchsize = static_cast<int>(inputMeta.NumSamples());
    int channels = static_cast<int>(inputMeta.NumChannels());
    int sh = static_cast<int>(inputMeta.Height());
    int sw = static_cast<int>(inputMeta.Width());

    std::vector<int> iw0(tw);
    std::vector<int> iw1(tw);
    std::vector<int> iw2(tw);
    std::vector<int> iw3(tw);
    std::vector<T4<T>> scaleX(tw);

    T widthScale = static_cast<T>(sw) / static_cast<T>(rw);
    T heightScale = static_cast<T>(sh) / static_cast<T>(rh);

    for (int ow = 0; ow < tw; ++ow) {
        int iw = cropOffsetX + ow;
        T fw = (iw + CENTER_ALIGN_PARAM) * widthScale - CENTER_ALIGN_PARAM;  // 目标像素点在源图对应点水平坐标
        int fw1 = std::min(static_cast<int>(floorf(fw)), sw - 1);  // 向下取整并防超出源图范围
        T lambda = std::min(fw - fw1, static_cast<T>(1));  // 映射点到取整点的距离
        iw0[ow] = Clip(fw1 - 1, 0, sw);  // 水平坐标分别取左1，右1，右2和fw1当前所在坐标,防止越界
        iw1[ow] = Clip(fw1, 0, sw);
        iw2[ow] = Clip(fw1 + 1, 0, sw);
        iw3[ow] = Clip(fw1 + 2, 0, sw);

        InterpolateCubic(lambda, scaleX[ow].values.data());
    }
    auto srcPtr = src.RawDataPtr<T>();
    auto dstPtr = dst.RawDataPtr<T>();

    int nts = 0;
    Balance::Task range = {0, 0};
    errCode = ws.GetNumThreads(nts);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get threads number", errCode);
    for (int t = 0; t < nts; t++) {
        errCode = Balance::Assign(batchsize * channels, nts, t, range);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to distribute tasks.", errCode);
        if (range.begin >= range.end) {
            break;
        }

        auto task = [range, sw, sh, tw, th, channels, srcPtr, dstPtr, cropOffsetY, heightScale, scaleX, iw0, iw1, iw2,
                iw3](int id, AccDataErrorCode &errCode) {
            CalcPixBicubic({sw, sh, tw, th, cropOffsetY, heightScale, channels, range}, {iw0, iw1, iw2, iw3}, scaleX,
                           srcPtr, dstPtr, errCode);
        };

        ws.GetThreadPool().AddTask(task);
    }

    errCode = ws.GetThreadPool().RunAll();
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to run task",
                                   AccDataErrorCode::H_SINGLEOP_ERROR);

    return AccDataErrorCode::H_OK;
}

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_TORCH_KERNEL_H_
