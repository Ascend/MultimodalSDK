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
 * Description: Source file for operator base check.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/tensor/OpsBaseChecker.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>
#include "acc/utils/LogImpl.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/core/framework/OperatorIndex.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace {
using namespace Acc;
constexpr size_t MIN_WIDTH = 10;
constexpr size_t MIN_HEIGHT = 10;
constexpr size_t MAX_WIDTH = 8192;
constexpr size_t MAX_HEIGHT = 8192;
const std::unordered_map<TensorFormat, std::unordered_map<std::string, int>> FORMAT_TO_INDEX_MAP = {
    {TensorFormat::NHWC, {{"batch", 0}, {"height", 1}, {"width", 2}, {"channel", 3}}},
    {TensorFormat::NCHW, {{"batch", 0}, {"height", 2}, {"width", 3}, {"channel", 1}}}};

const TensorConstraint BASIC_TENSOR_CONSTRAINT = {"cpu",
                                                  {DataType::UINT8},
                                                  {TensorFormat::NHWC},
                                                  {{"batch", EnumeratedConstraint{{1}}},
                                                   {"height", RangeConstraint{MIN_HEIGHT, MAX_HEIGHT}},
                                                   {"width", RangeConstraint{MIN_WIDTH, MAX_WIDTH}},
                                                   {"channel", EnumeratedConstraint{{3}}}}};

const TensorConstraint NORMALIZE_TENSOR_CONSTRAINT_CPU = {
    "cpu",
    {DataType::FLOAT32},
    {TensorFormat::NHWC, TensorFormat::NCHW},
    {{"batch", EnumeratedConstraint{{1}}}, {"channel", EnumeratedConstraint{{3}}}}};

const TensorConstraint BASIC_QWENFUSION_CONSTRAINT = {"cpu",
                                                      {DataType::UINT8},
                                                      {TensorFormat::NHWC},
                                                      {{"height", RangeConstraint{MIN_HEIGHT, MAX_HEIGHT}},
                                                       {"width", RangeConstraint{MIN_WIDTH, MAX_WIDTH}},
                                                       {"channel", EnumeratedConstraint{{3}}}}};

const TensorConstraint TO_TENSOR_INPUT_TENSOR_CONSTRAINT_CPU = {
    "cpu",
    {DataType::UINT8},
    {TensorFormat::NHWC, TensorFormat::NCHW},
    {{"batch", EnumeratedConstraint{{1}}}, {"channel", EnumeratedConstraint{{3}}}}};

const TensorConstraint TO_TENSOR_OUTPUT_TENSOR_CONSTRAINT_CPU = {
    "cpu",
    {DataType::FLOAT32},
    {TensorFormat::NHWC, TensorFormat::NCHW},
    {{"batch", EnumeratedConstraint{{1}}}, {"channel", EnumeratedConstraint{{3}}}}};

// resize constraint
const OperatorTensorConstraints CPU_RESIZE_CONSTRAINT{{BASIC_TENSOR_CONSTRAINT}, {BASIC_TENSOR_CONSTRAINT}};

// crop constraint
const OperatorTensorConstraints CPU_CROP_CONSTRAINT{{BASIC_TENSOR_CONSTRAINT}, {BASIC_TENSOR_CONSTRAINT}};

// normalize constraint
const OperatorTensorConstraints CPU_NORMALIZE_CONSTRAINT{{NORMALIZE_TENSOR_CONSTRAINT_CPU},
                                                         {NORMALIZE_TENSOR_CONSTRAINT_CPU}};

// QwenFusion constraint
const OperatorTensorConstraints CPU_QWENFUSION_CONSTRAINT{{BASIC_QWENFUSION_CONSTRAINT}, {BASIC_QWENFUSION_CONSTRAINT}};

// normalize constraint
const OperatorTensorConstraints CPU_TO_TENSOR_CONSTRAINT{{TO_TENSOR_INPUT_TENSOR_CONSTRAINT_CPU},
                                                         {TO_TENSOR_OUTPUT_TENSOR_CONSTRAINT_CPU}};

// all operators configs for single tensor check
const std::unordered_map<OperatorId, OperatorTensorConstraints> OPERATOR_CONSTRAINT_MAP = {
    {OperatorId::CROP, CPU_RESIZE_CONSTRAINT},
    {OperatorId::RESIZE, CPU_CROP_CONSTRAINT},
    {OperatorId::NORMALIZE, CPU_NORMALIZE_CONSTRAINT},
    {OperatorId::QWENFUSION, CPU_QWENFUSION_CONSTRAINT},
    {OperatorId::TOTENSOR, CPU_TO_TENSOR_CONSTRAINT}};

std::string DataTypeToString(DataType dt)
{
    switch (dt) {
        case DataType::INT8:
            return "INT8";
        case DataType::UINT8:
            return "UINT8";
        case DataType::FLOAT32:
            return "FLOAT32";
        default:
            return "UNKNOWN";
    }
}

std::string GetSupportedDataTypesString(const std::vector<DataType>& types)
{
    std::ostringstream oss;
    for (size_t i = 0; i < types.size(); ++i) {
        oss << DataTypeToString(types[i]);
        if (i != types.size() - 1) {
            oss << ", ";
        }
    }
    return oss.str();
}
std::string TensorFormatToString(TensorFormat fmt)
{
    switch (fmt) {
        case TensorFormat::NCHW:
            return "NCHW";
        case TensorFormat::NHWC:
            return "NHWC";
        case TensorFormat::ND:
            return "ND";
        default:
            return "UNKNOWN";
    }
}

std::string GetSupportedFormatsString(const std::vector<TensorFormat>& formats)
{
    std::ostringstream oss;
    for (size_t i = 0; i < formats.size(); ++i) {
        oss << TensorFormatToString(formats[i]);
        if (i != formats.size() - 1) {
            oss << ", ";
        }
    }
    return oss.str();
}
std::string GetEnumeratedConstraintString(const std::vector<int>& values)
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        oss << values[i];
        if (i != values.size() - 1) {
            oss << ", ";
        }
    }
    return oss.str();
}
} // namespace
namespace Acc {
OpsBaseChecker::OpsBaseChecker(const OperatorId& opId) : opId_(opId)
{
    auto iter = OPERATOR_CONSTRAINT_MAP.find(opId);
    if (iter == OPERATOR_CONSTRAINT_MAP.end()) {
        throw std::runtime_error("Operator constraint is not found, please check the OPERATOR_CONSTRAINT_MAP.");
    }
}

ErrorCode OpsBaseChecker::CheckAndImplicitMalloc(const OperatorContext& ctx)
{
    outputMallocFlags_ = std::vector<bool>(ctx.outputTensorRefs.size(), true);
    for (size_t i = 0; i < ctx.outputTensorRefs.size(); i++) {
        if (ctx.outputTensorRefs[i].get().Ptr() == nullptr) {
            outputMallocFlags_[i] = false;
            continue;
        }
    }
    ErrorCode ret = CheckEachTensorValid(ctx.inputTensorRefs, ctx.outputTensorRefs);
    if (ret != SUCCESS) {
        return ret;
    }
    ret = CheckCustomRules(ctx);
    if (ret != SUCCESS) {
        return ret;
    }
    ret = ImplicitMalloc(ctx);
    return ret;
}

ErrorCode OpsBaseChecker::CheckTensorDimension(
    const Tensor& tensor, const std::unordered_map<std::string, DimensionConstraint>& dimConstraints) const
{
    // No check will be performed if the tensor format is not NCHW or NHWC
    if (tensor.Format() != TensorFormat::NCHW && tensor.Format() != TensorFormat::NHWC) {
        return SUCCESS;
    }
    auto tensorShape = tensor.Shape();
    for (const auto& kv : dimConstraints) {
        const std::string& dimType = kv.first;
        const DimensionConstraint& dimConstraint = kv.second;
        auto formatIt = FORMAT_TO_INDEX_MAP.find(tensor.Format());
        const auto& innerMap = formatIt->second;
        auto dimIt = innerMap.find(dimType);
        auto index = dimIt->second;
        if (const RangeConstraint* range = std::get_if<RangeConstraint>(&dimConstraint)) {
            // Note: No check will be performed if the range value remains the default (-1)
            if (range->minVal != -1 && static_cast<int>(tensorShape[index]) < range->minVal) {
                LogError << "The value of dimension " << dimType << " is " << tensorShape[index]
                         << ", which should not be lower than " << range->minVal << "."
                         << GetErrorInfo(ERR_INVALID_PARAM);
                return ERR_INVALID_PARAM;
            }
            if (range->maxVal != -1 && static_cast<int>(tensorShape[index]) > range->maxVal) {
                LogError << "The value of dimension " << dimType << " is " << tensorShape[index]
                         << ", which should not be larger than " << range->maxVal << "."
                         << GetErrorInfo(ERR_INVALID_PARAM);
                return ERR_INVALID_PARAM;
            }
        } else if (const EnumeratedConstraint* enumerated = std::get_if<EnumeratedConstraint>(&dimConstraint)) {
            // Note: No check will be performed if the enumerate value remains the default (empty)
            bool notExisted = (std::find(enumerated->values.begin(), enumerated->values.end(),
                                         static_cast<int>(tensorShape[index])) == enumerated->values.end());
            if (enumerated->values.size() != 0 && notExisted) {
                LogError << "The value of dimension " << dimType << " is " << tensorShape[index] << ", which should be "
                         << GetEnumeratedConstraintString(enumerated->values) << "." << GetErrorInfo(ERR_INVALID_PARAM);
                return ERR_INVALID_PARAM;
            }
        } else {
            LogDebug << "Unexpected DimensionConstraint type encountered." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    return SUCCESS;
}

ErrorCode OpsBaseChecker::CheckTensorAttributes(const Tensor& tensor, const TensorConstraint& tensorConstraint) const
{
    if (std::strcmp(tensor.Device().get(), tensorConstraint.device.c_str()) != 0) {
        LogError << "The device of current input/output is not required. Current device is " << tensor.Device().get()
                 << ", but the expected device is " << tensorConstraint.device.c_str() << "."
                 << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (std::find(tensorConstraint.dataTypes.begin(), tensorConstraint.dataTypes.end(), tensor.DType()) ==
        tensorConstraint.dataTypes.end()) {
        LogError << "The datatype of current input/output is not required. Current datatype is "
                 << DataTypeToString(tensor.DType()) << ", but the expected datatype is "
                 << GetSupportedDataTypesString(tensorConstraint.dataTypes) << "." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (std::find(tensorConstraint.formats.begin(), tensorConstraint.formats.end(), tensor.Format()) ==
        tensorConstraint.formats.end()) {
        LogError << "The format of current input/output is not required. Current format is "
                 << TensorFormatToString(tensor.Format()) << ", but the expected format is "
                 << GetSupportedFormatsString(tensorConstraint.formats) << "." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (CheckTensorDimension(tensor, tensorConstraint.dimensionConstraints) != SUCCESS) {
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode OpsBaseChecker::CheckEachTensorValid(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                                               const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs)
{
    auto opTensorConstraint = OPERATOR_CONSTRAINT_MAP.at(opId_);
    for (size_t i = 0; i < opTensorConstraint.inputConstraints.size(); i++) {
        if (CheckTensorAttributes(inputTensorRefs[i].get(), opTensorConstraint.inputConstraints[0]) != SUCCESS) {
            LogError << "Check input attributes failed." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    for (size_t i = 0; i < opTensorConstraint.outputConstraints.size(); i++) {
        if (!outputMallocFlags_[i]) {
            continue;
        }
        if (CheckTensorAttributes(outputTensorRefs[i].get(), opTensorConstraint.outputConstraints[0]) != SUCCESS) {
            LogError << "Check output attributes failed." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    return SUCCESS;
}

ErrorCode OpsBaseChecker::CheckMultiTensorMatch(
    const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
    const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs)
{
    auto tensorShape = inputTensorRefs[0].get().Shape();
    auto tensorDtype = inputTensorRefs[0].get().DType();
    auto tensorFormat = inputTensorRefs[0].get().Format();
    for (size_t i = 1; i < inputTensorRefs.size(); i++) {
        if (tensorShape != inputTensorRefs[i].get().Shape()) {
            LogError << "The shape of inputs should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        if (tensorDtype != inputTensorRefs[i].get().DType()) {
            LogError << "The datatype of inputs should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        if (tensorFormat != inputTensorRefs[i].get().Format()) {
            LogError << "The format of inputs should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    for (size_t i = 0; i < outputTensorRefs.size(); i++) {
        if (!outputMallocFlags_[i]) {
            continue;
        }
        if (tensorShape != outputTensorRefs[i].get().Shape()) {
            LogError << "The shape of inputs and outputs should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        if (tensorDtype != outputTensorRefs[i].get().DType()) {
            LogError << "The datatype of inputs and outputs should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        if (tensorFormat != outputTensorRefs[i].get().Format()) {
            LogError << "The format of inputs and outputs should be the same." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
    }
    return SUCCESS;
}

ErrorCode OpsBaseChecker::CheckCustomRules(const OperatorContext& ctx)
{
    if (CheckMultiTensorMatch(ctx.inputTensorRefs, ctx.outputTensorRefs) != SUCCESS) {
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode OpsBaseChecker::ImplicitMalloc(const OperatorContext& ctx)
{
    auto inferShape = ctx.inputTensorRefs[0].get().Shape();
    auto inferDatatype = ctx.inputTensorRefs[0].get().DType();
    auto inferFromat = ctx.inputTensorRefs[0].get().Format();
    for (size_t i = 0; i < ctx.outputTensorRefs.size(); i++) {
        if (outputMallocFlags_[i]) {
            continue;
        }
        auto totalBytes = ctx.inputTensorRefs[0].get().AuxInfo().totalBytes;
        if (totalBytes == 0) {
            LogError << "The data size of input should not be 0." << GetErrorInfo(ERR_INVALID_PARAM);
            return ERR_INVALID_PARAM;
        }
        char* data = new(std::nothrow) char[totalBytes];
        if (data == nullptr) {
            LogError << "Failed to malloc for tensor." << GetErrorInfo(ERR_BAD_ALLOC);
            return ERR_BAD_ALLOC;
        }
        std::shared_ptr<void> dstPtr(static_cast<void*>(data), [](void* ptr) { delete[] static_cast<char*>(ptr); });
        ctx.outputTensorRefs[i].get() =
            Tensor(dstPtr, inferShape, inferDatatype, inferFromat, ctx.inputTensorRefs[0].get().Device().get());
    }
    return SUCCESS;
}

} // namespace Acc