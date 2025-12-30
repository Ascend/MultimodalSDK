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
 * Description: preprocess file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef PY_PYPREPROCESS_H
#define PY_PYPREPROCESS_H
#include <vector>

#include "PyUtil.h"
#include "PyImage.h"
#include "PyTensor.h"

namespace PyAcc {
class Qwen2VLProcessor {
public:
    /**
     * @brief Python interface entry for preprocessing
     *
     * Performs Resize + ToTensor + Normalize on the input images from Python.
     *
     * @param pyImages Input images from Python
     * @param mean Normalize mean vector (length = 3)
     * @param std Normalize standard deviation vector (length = 3)
     * @param resizeW Resize width
     * @param resizeH Resize height
     * @return std::vector<Tensor> Output tensors after preprocessing
     */
    static std::vector<Tensor> Preprocess(const std::vector<Image>& pyImages, const std::vector<float>& mean,
                                          const std::vector<float>& std, int resizeW, int resizeH);
};
} // namespace PyAcc
#endif // PY_PYPREPROCESS_H
