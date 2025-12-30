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
#include "to_tensor_args.h"

#include "tensor/tensor_image.h"

namespace acclib {
namespace accdata {

    AccDataErrorCode ToTensorArgs::Setup(const OpSpec &spec, Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    int64_t layout = 0;
    errCode = spec.GetArg<int64_t>("layout", layout);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get arguments.", errCode);

    mLayout = static_cast<TensorLayout>(layout);
    if (mLayout != TensorLayout::NCHW && mLayout != TensorLayout::NHWC) {
        ACCDATA_ERROR("The output layout should be NHWC or NCHW.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    auto &inputTensorList = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to get input.", errCode);

    if (inputTensorList.IsEmpty() || !inputTensorList.IsValid()) {
        ACCDATA_ERROR("Illegal tensor.");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }

    if (inputTensorList[0].DataType() != TensorDataType::UINT8) {
        ACCDATA_ERROR("Input of ToTensor should not be empty and the datatype of ToTensor input should be uint8.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    if (inputTensorList[0].Layout() != TensorLayout::NCHW && inputTensorList[0].Layout() != TensorLayout::NHWC) {
        ACCDATA_ERROR("The input layout should be NHWC or NCHW.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    mSameLayout = (mLayout == inputTensorList[0].Layout());

    return AccDataErrorCode::H_OK;
}

} // namespace accdata
} // namespace acclib
