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
 * Description: Definition of TensorUtils.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef TENSORUTILS_H
#define TENSORUTILS_H

#include "accdata_tensor.h"
#include "acc/tensor/TensorDataType.h"

namespace Acc {
    /**
     * @brief Convert TensorFormat to TensorLayout
     * @param format TensorFormat to be converted
     * @return TensorLayout
     */
    acclib::accdata::TensorLayout ToTensorLayout(TensorFormat format);

    /**
     * @brief Convert TensorLayout to TensorFormat
     * @param layout TensorLayout to be converted
     * @return TensorFormat
     */
    TensorFormat ToTensorFormat(acclib::accdata::TensorLayout layout);

    /**
     * @brief Convert DataType to TensorDataType
     * @param dataType DataType to be converted
     * @return TensorDataType
     */
    acclib::accdata::TensorDataType ToTensorDataType(DataType dataType);

    /**
     * @brief Convert TensorDataType to DataType
     * @param tensorDataType TensorDataType to be converted
     * @return DataType
     */
    DataType ToDataType(acclib::accdata::TensorDataType tensorDataType);
}

#endif // TENSORUTILS_H
