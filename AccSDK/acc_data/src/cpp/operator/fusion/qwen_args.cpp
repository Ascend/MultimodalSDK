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

#include "qwen_args.h"

namespace acclib {
namespace accdata {

AccDataErrorCode QwenArgs::Setup(const OpSpec &spec, Workspace &ws)
{
    AccDataErrorCode errCode = AccDataErrorCode::H_OK;
    if (spec.HasArg("min_pixels")) {
        errCode = spec.GetArg<int64_t>("min_pixels", ws, mMinPixels);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    }

    if (spec.HasArg("max_pixels")) {
        errCode = spec.GetArg<int64_t>("max_pixels", ws, mMaxPixels);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    }

    if (mMinPixels < RESIZE_HEIGHT_MIN * RESIZE_WIDTH_MIN || mMaxPixels > RESIZE_HEIGHT_MAX * RESIZE_WIDTH_MAX
        || mMinPixels > mMaxPixels) {
        ACCDATA_ERROR("Min pixels and max pixels should in [" << RESIZE_WIDTH_MIN * RESIZE_HEIGHT_MIN <<
            ", " << RESIZE_HEIGHT_MAX * RESIZE_WIDTH_MAX << "], " <<
            "and min pixels should be less than or equal to max pixels.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    if (spec.HasArg("patch_size")) {
        errCode = spec.GetArg<int64_t>("patch_size", ws, mPatchSize);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    }

    if (spec.HasArg("temporal_patch_size")) {
        errCode = spec.GetArg<int64_t>("temporal_patch_size", ws, mTemporalPatchSize);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    }

    if (mTemporalPatchSize != 2LL) {
        ACCDATA_ERROR("Temporal patch size should eq 2.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    if (spec.HasArg("merge_size")) {
        errCode = spec.GetArg<int64_t>("merge_size", ws, mMergeSize);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get argument.", errCode);
    }

    if (mPatchSize <= 0 || mMergeSize <= 0) {
        ACCDATA_ERROR("Patch size and merge size should be greater than 0.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get input.", errCode);
    if (input.NumTensors() < 1 || input[0].DataType() != TensorDataType::UINT8) {
        ACCDATA_ERROR("Input for Qwen2-VL should not be empty and the datatype should be uint8 and not empty.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }
    mNeedRepeat = (input[0].Shape()[0] == 1);

    ACCDATA_DEBUG("Qwen args: Min pixels = " << mMinPixels << ", Max pixels = " << mMaxPixels << ", Patch size = " <<
        mPatchSize << ", Temporal patch size = " << mTemporalPatchSize << ", Merge size = " << mMergeSize <<
        " Need repeat = " << (mNeedRepeat ? "yes" : "no") << ".");
    return errCode;
}

}  // namespace accdata
}  // namespace acclib