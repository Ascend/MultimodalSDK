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
#include "external_source.h"

#include "operator/op_factory.h"

namespace acclib {
namespace accdata {

AccDataErrorCode ExternalSource::Run(Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto &spec = GetSpec();
    if (ws.NumOutput() != spec.NumOutput()) {
        ACCDATA_ERROR("The number of outputs is inconsistent.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
    if (mInputs.Empty()) {
        ACCDATA_ERROR("ExternalSource is empty.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
    /* Swap the external input to the output. */
    std::list<TensorList> input;
    errCode = mInputs.PopFront(input);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to pop front.", errCode);

    auto &output = ws.GetOutput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);

    std::swap(output, input.front());
    mInputs.Recycle(input);
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ExternalSource::Feed(std::shared_ptr<TensorList> data, bool copy)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto input = mInputs.GetFree();
    if (copy) {
        errCode = input.front().Copy(data);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to copy.", errCode);
    } else {
        errCode = input.front().ShareData(data);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to share data.", errCode);
    }
    mInputs.PushBack(input);
    return AccDataErrorCode::H_OK;
} // namespace accdata
} // namespace acclib

ACCDATA_REGISTER_OPERATOR(ExternalSource, acclib::accdata::ExternalSource);

}
