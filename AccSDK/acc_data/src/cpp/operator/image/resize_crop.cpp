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
 * @Date: 2025-2-17 16:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-17 16:00:00
 */
#include "resize_crop.h"

#include "operator/op_factory.h"
#include "resize_torch_kernel.h"
#include "common/tracer.h"

namespace acclib {
namespace accdata {

AccDataErrorCode ResizeCrop::Run(Workspace &ws)
{
    TRACE_BEGIN(resize_crop)
    auto errCode = AccDataErrorCode::H_OK;
    ACCDATA_DEBUG("ResizeCrop run.");

    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Input out of range.", errCode);

    auto &output = ws.GetOutput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Output out of range.", errCode);

    if (input.IsEmpty() || !input.IsValid()) {
        ACCDATA_ERROR("Illegal input.");
        return AccDataErrorCode::H_COMMON_INVALID_PARAM;
    }

    errCode = Setup(ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the resize crop operator.",
                                   errCode);

    if (input.NumTensors() > output.NumTensors()) {
        ACCDATA_ERROR("The number of input tensors should not exceed that of output.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }

    switch (input[0].DataType()) {
        case TensorDataType::FP32:
            if (input[0].Layout() != TensorLayout::NCHW) {
                ACCDATA_ERROR("The ResizeCrop operator only support NCHW layout while input type is fp32.");
                return AccDataErrorCode::H_SINGLEOP_ERROR;
            }
            return TorchResizeCrop(ws);
        default:
            ACCDATA_ERROR("Resize_crop only support FP32 now.");
            return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
    TRACE_END(resize_crop)

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ResizeCrop::Setup(Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto &spec = GetSpec();
    if (ws.NumOutput() != spec.NumOutput()) {
        ACCDATA_ERROR("The number of outputs is inconsistent.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }

    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK,
        "Input out of range during fused operation execution.", errCode);
    if (input.NumTensors() < 1) {
        ACCDATA_ERROR("The number of input is empty.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    errCode = mInputMeta.Setup(input[0]);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the input meta arguments.",
                                   errCode);

    errCode = mResizeArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the resize arguments.",
                                   errCode);

    errCode = mCropArgs.Setup(spec, ws);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to set up the crop arguments.", errCode);

    if (mCropArgs.Width() > mResizeArgs.Width() || mCropArgs.Height() > mResizeArgs.Height()) {
        ACCDATA_ERROR("Crop size can not be greater than resize size!!");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ResizeCrop::TorchResizeCrop(Workspace &ws)
{
    auto errCode = AccDataErrorCode::H_OK;
    auto &input = ws.GetInput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);

    int numTensors = input.NumTensors();
    auto outTensorShape = input[0].Shape(); // NCHW
    constexpr int torchTensorHeightDim = 2;
    constexpr int torchTensorWidthDim = 3;
    outTensorShape[torchTensorHeightDim] = mCropArgs.Height(); // output height
    outTensorShape[torchTensorWidthDim] = mCropArgs.Width();   // output width
    TensorListShape shape(numTensors, outTensorShape);
    auto &output = ws.GetOutput(0, errCode);
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Out of range.", errCode);

    errCode = output.Resize(shape, TensorDataTypeView(input.Tensors()));
    ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to resize.", errCode);
    for (int i = 0; i < numTensors; ++i) {
        auto &in = input[i];
        auto &out = output[i];
        ACCDATA_DEBUG("Resize crop for " << i << " size [" << mCropArgs.Height() << ", " << mCropArgs.Width() << "]");
        errCode = TorchFloatImpl(out, in, ws);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to execute torch float implement.",
                                       errCode);
    }
    output.SetLayout(TensorLayout::NCHW);

    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ResizeCrop::TorchFloatImpl(Tensor &result, const Tensor &input, Workspace &ws)
{
    auto shape = input.Shape();

    int targetH = mCropArgs.Height();
    int targetW = mCropArgs.Width();
    int resizeH = mResizeArgs.Height();
    int resizeW = mResizeArgs.Width();

    if (mResizeArgs.Mode() == InterpMode::BILINEAR) {
        return TorchBilinear<float>(input, result, resizeH, resizeW, targetH, targetW, mInputMeta, ws);
    } else if (mResizeArgs.Mode() == InterpMode::BICUBIC) {
        return TorchBicubic<float>(input, result, resizeH, resizeW, targetH, targetW, mInputMeta, ws);
    } else {
        ACCDATA_ERROR("ResizeCrop only support bilinear and bicubic interpolation mode now.");
        return AccDataErrorCode::H_SINGLEOP_ERROR;
    }
    return ws.GetThreadPool().WaitAll();
}

ACCDATA_REGISTER_OPERATOR(ResizeCrop, ResizeCrop);

} // namespace accdata
} // namespace acclib