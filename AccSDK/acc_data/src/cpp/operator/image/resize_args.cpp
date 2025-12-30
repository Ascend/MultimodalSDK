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
#include "resize_args.h"

namespace acclib {
namespace accdata {
AccDataErrorCode ResizeArgs::Setup(const OpSpec &spec, Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    std::vector<int64_t> resize;
    errCode = spec.GetArg<std::vector<int64_t>>("resize", ws, resize);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    /* 2 indicates {height, width} */
    ACCDATA_CHECK_ERRORCODE_RETURN(resize.size() == 2, "'resize' argument should have 2 elements [height, width].",
        AccDataErrorCode::H_COMMON_OPERATOR_ERROR);

    mResizeHeight = resize[0];
    mResizeWidth = resize[1];

    if (mResizeHeight < RESIZE_HEIGHT_MIN || mResizeWidth < RESIZE_WIDTH_MIN || mResizeHeight > RESIZE_HEIGHT_MAX ||
        mResizeWidth > RESIZE_WIDTH_MAX) {
        ACCDATA_ERROR("Resize size should be within " << RESIZE_HEIGHT_MIN << " * " << RESIZE_WIDTH_MIN << " to " <<
            RESIZE_HEIGHT_MAX << " * " << RESIZE_WIDTH_MAX << ", Input is (" << mResizeHeight << ", " << mResizeWidth <<
            ")");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    if (spec.HasArg("interpolation_mode")) {
        std::string interMode;
        errCode = spec.GetArg<std::string>("interpolation_mode", interMode);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
        if (interMode == "bilinear") {
            mInterMode = InterpMode::BILINEAR;
        } else if (interMode == "bicubic") {
            mInterMode = InterpMode::BICUBIC;
        } else {
            ACCDATA_ERROR("Unknown interpolation mode.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
    }
    ACCDATA_DEBUG("Resize args: W = " << mResizeWidth << ", H = " << mResizeHeight << ", InterMode = '" << mInterMode <<
        "'.");
    return AccDataErrorCode::H_OK;
}
} // namespace accdata
} // namespace acclib
