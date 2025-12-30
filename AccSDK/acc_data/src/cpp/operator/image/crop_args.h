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
 * @Date: 2025-2-10 19:00:00
 * @LastEditTime: 2025-2-10 19:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_IMAGE_CROP_ARGS_H_
#define ACCDATA_SRC_CPP_OPERATOR_IMAGE_CROP_ARGS_H_

#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"

namespace acclib {
namespace accdata {

/**
 * @brief Arguments for crop operation.
 *
 * Arguments that can be specified through OpSpec:
 * Required arguments:
 * - crop_pos_x: Normalized (0.0 - 1.0) horizontal position of the start of the cropping window.
 * The actual position is calculated as 'x = crop_pos_x * (srcWidth - crop_w)'.
 * - crop_pos_y: Normalized (0.0 - 1.0) vertical position of the start of the cropping window.
 * The actual position is calculated as 'y = crop_pos_y * (srcHeight - crop_h)'
 * - crop: A vector contains [crop_w, crop_h], it's incompatible with 'crop_w' and 'crop_h',
 * and has higher priority.
 * - crop_w: Cropping the window width(in pixels).
 * - crop_h: Cropping the window height(in pixels).
 * Optional arguments:
 * - round_mode: 'round' or 'truncate', default is 'round'.
 */
class CropArgs {
public:
    CropArgs() = default;
    ~CropArgs() = default;

    /* *
     * @brief Prepare arguments from OpSpec and Workspace.
     * @note Must be called before other member functions.
     */
    AccDataErrorCode Setup(const OpSpec& spec, Workspace& ws);

    /* * @brief Cropping the window width. */
    int64_t Width()
    {
        return mCropWidth;
    }

    /* * @brief Cropping the window height. */
    int64_t Height()
    {
        return mCropHeight;
    }

    /* *
     * @brief Get vertical position of the start of the cropping window.
     *
     * @param height  Height of original image
     */
    int64_t Top(int64_t height)
    {
        return std::max(0L, mRound(mCropPosY * (height - mCropHeight)));
    }

    /* *
     * @brief Get horizontal position of the start of the cropping window.
     *
     * @param height  Width of original image
     */
    int64_t Left(int64_t width)
    {
        return std::max(0L, mRound(mCropPosX * (width - mCropWidth)));
    }

private:
    float mCropPosX{ 0.5 };
    float mCropPosY{ 0.5 };
    int64_t mCropWidth{ 0 };
    int64_t mCropHeight{ 0 };
    std::function<int64_t(double)> mRound = [](double v) {
        return static_cast<int64_t>(std::round(v));
    };
};

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_IMAGE_CROP_ARGS_H_
