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

#ifndef ACCDATA_OPERATOR_IMAGE_QWEN_ARGS_H
#define ACCDATA_OPERATOR_IMAGE_QWEN_ARGS_H

#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"

/**
 * @brief Arguments for qwen2-vl.
 *
 * Arguments that can be specified through OpSpec:
 * Required arguments:
 * - min_pixels: The min pixels of the image to resize the image, defaults to `56 * 56`
 * - max_pixels: The max pixels of the image to resize the image, defaults to `28 * 28 * 1280`
 * - patch_size: The spacial patch size of the vision encoder, defaults to 14
 * - temporal_patch_size: The temporal patch size of the vision encoder, defaults to 2
 * - merge_size: The merge size of the vision encoder to llm encoder, defaults to 2
 */
namespace acclib {
namespace accdata {

class QwenArgs {
public:
    QwenArgs() = default;
    ~QwenArgs() = default;

    /**
     * @brief Prepare arguments from OpSpec and Workspace.
     * @note Must be called before other member functions.
     */
    AccDataErrorCode Setup(const OpSpec &spec, Workspace &ws);

    /** Getter method */
    inline int64_t MinPixels() const
    {
        return mMinPixels;
    }

    inline int64_t MaxPixels() const
    {
        return mMaxPixels;
    }

    inline int64_t PatchSize() const
    {
        return mPatchSize;
    }

    inline int64_t TemporalPatchSize() const
    {
        return mTemporalPatchSize;
    }

    inline int64_t MergeSize() const
    {
        return mMergeSize;
    }

    inline bool NeedRepeat() const
    {
        return mNeedRepeat;
    }

private:
    int64_t mMinPixels = 56 * 56;
    int64_t mMaxPixels = 28 * 28 * 1280;
    int64_t mPatchSize = 14;
    int64_t mTemporalPatchSize = 2;
    int64_t mMergeSize = 2;
    bool mNeedRepeat = false;
};

}  // namespace accdata
}  // namespace acclib

#endif // ACCDATA_SRC_CPP_OPERATOR_FUSION_QWEN_ARGS_H_
