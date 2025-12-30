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
 * @Date: 2025-2-17 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-17 9:00:00
 */

#include "crop_args.h"

namespace acclib {
namespace accdata {
AccDataErrorCode CropArgs::Setup(const OpSpec &spec, Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    (void)spec.GetArg<float>("crop_pos_x", ws, mCropPosX);

    (void)spec.GetArg<float>("crop_pos_y", ws, mCropPosY);

    if (mCropPosX < 0.0 || mCropPosX > 1.0 || mCropPosY < 0.0 || mCropPosY > 1.0) {
        ACCDATA_ERROR("crop_pos_x(" << mCropPosX << ") or crop_pos_y(" << mCropPosY << ") is out of range [0, 1]");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    std::vector<int64_t> crop;
    errCode = spec.GetArg<std::vector<int64_t>>("crop", ws, crop);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    /* 2 indicates {height, width} */
    ACCDATA_CHECK_ERRORCODE_RETURN(crop.size() == 2, "'crop' argument should have 2 elements [height, width].",
        AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    mCropHeight = crop[0];
    mCropWidth = crop[1];

    if (mCropHeight < CROP_HEIGHT_MIN || mCropWidth < CROP_WIDTH_MIN || mCropHeight > CROP_HEIGHT_MAX ||
        mCropWidth > CROP_WIDTH_MAX) {
        ACCDATA_ERROR("Crop size should be within " << CROP_HEIGHT_MIN << " * " << CROP_WIDTH_MIN << " to " <<
            CROP_HEIGHT_MAX << " * " << CROP_WIDTH_MAX << "!! (height * width)");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    std::string roundMode = "round";
    if (spec.HasArgOrArgInput("round_mode")) {
        errCode = spec.GetArg<std::string>("round_mode", ws, roundMode);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
        if (roundMode == "round") {
            /* default is round. */
        } else if (roundMode == "truncate") {
            mRound = [](double v) { return static_cast<int64_t>(v); };
        } else {
            ACCDATA_ERROR("Unsupported round mode.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
    }
    ACCDATA_DEBUG("Crop args: PosX = " << mCropPosX << ", PosY = " << mCropPosY << ", W = " << mCropWidth << ", H = " <<
        mCropHeight << ", round mode = '" << roundMode << "'.";);

    return AccDataErrorCode::H_OK;
}
} // namespace accdata
} // namespace acclib
