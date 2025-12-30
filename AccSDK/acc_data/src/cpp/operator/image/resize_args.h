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
 * @Date: 2025-2-10 20:00:00
 * @LastEditTime: 2025-2-10 20:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_ARGS_H_
#define ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_ARGS_H_

#include "operator/op_spec.h"
#include "pipeline/workspace/workspace.h"

namespace acclib {
namespace accdata {

/**
 * @brief Interpolation mode
 */
enum class InterpMode {
    BILINEAR,
    BICUBIC,
    LAST,
};

inline std::ostream& operator<<(std::ostream& os, InterpMode v)
{
    switch (v) {
        case InterpMode::BILINEAR:
            os << "BILINEAR"; break;
        case InterpMode::BICUBIC:
            os << "BICUBIC"; break;
        default:
            os << "Unknown"; break;
    }
    return os;
}

/**
 * @brief Arguments for resize operation.
 *
 * Arguments that can be specified through OpSpec:
 * Required argument:
 * - resize_w: The width of the resized image.
 * - resize_h: The height of the resized image.
 * Optional argument:
 * - interpolation_mode: Interpolation mode. Default is 'bilinear'.
 */
class ResizeArgs {
public:
    ResizeArgs() = default;
    ~ResizeArgs() = default;

    /* *
     * @brief Prepare arguments from OpSpec and Workspace.
     * @note Must be called before other member functions.
     */
    AccDataErrorCode Setup(const OpSpec& spec, Workspace& ws);

    /* * @brief The width of the resized image. */
    int64_t Width()
    {
        return mResizeWidth;
    }

    /* * @brief The height of the resized image. */
    int64_t Height()
    {
        return mResizeHeight;
    }

    /* * @brief Interpolation mode. */
    InterpMode Mode()
    {
        return mInterMode;
    }

private:
    int64_t mResizeWidth{ 0 };
    int64_t mResizeHeight{ 0 };
    InterpMode mInterMode{ InterpMode::BILINEAR };
};

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_IMAGE_RESIZE_ARGS_H_
