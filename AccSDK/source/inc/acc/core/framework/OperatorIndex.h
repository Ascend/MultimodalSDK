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
 * Description: Index for support operators.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef OPERATOR_INDEX_H
#define OPERATOR_INDEX_H

namespace Acc {
    /**
     * @brief Operator ID enumeration
     *
     * Defines unique identifiers for different image processing operators.
     * These IDs are used as keys in the operator map to register and lookup
     * specific operator implementations at runtime.
     *
     */
    enum class OperatorId {
        RESIZE = 0,     // Image resizing operator - changes image dimensions
        CROP = 1,       // Image cropping operator - extracts rectangular region
        TOTENSOR,       // Image ToTensor operator - equivalent to torchvision.transforms.ToTensor
        NORMALIZE,      // Tensor normalization operator - scales pixel values to specified range
        QWENFUSION,     // QwenFusion operator - preprocess operation for Qwen2VL
        OTHER,
    };
}

#endif // OPERATOR_INDEX_H