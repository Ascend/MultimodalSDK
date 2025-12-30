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
#include "normalize_args.h"

namespace acclib {
namespace accdata {

AccDataErrorCode NormalizeArgs::Setup(const OpSpec &spec, Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    errCode = spec.GetArg<std::vector<float>>("mean", ws, mMean);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get the mean argument.", errCode);

    errCode = spec.GetArg<std::vector<float>>("stddev", ws, mStddev);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get the stddev argument.", errCode);

    if (mMean.size() != RGB_CHANNELS || mStddev.size() != RGB_CHANNELS) {
        ACCDATA_ERROR("The mean argument and the stddev argument should have 3 elements.");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }

    for (auto &m : mMean) {
        if (m < 0 || m > 1) {
            ACCDATA_ERROR("The mean argument must in [0, 1], mean is: " << m);
            return AccDataErrorCode::H_COMMON_INVALID_PARAM;
        }
    }

    float scale = 1;
    if (spec.HasArgOrArgInput("scale")) {
        errCode = spec.GetArg<float>("scale", ws, scale);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get the scale argument.", errCode);
    }

    ACCDATA_DEBUG("Normalize args: mean = " << mMean << ", stddev = " << mStddev << ", scale = "<< scale << ".");
    for (auto &s : mStddev) {
        ACCDATA_CHECK_ERRORCODE_RETURN(s > 0, "stddev must be larger than zero.",
            AccDataErrorCode::H_COMMON_INVALID_PARAM);
        s = 1 / s * scale;
    }

    return AccDataErrorCode::H_OK;
}

} // namespace accdata
} // namespace acclib
