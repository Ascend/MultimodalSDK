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
 * Description: Head file for operator base check.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef OPS_BASE_CHECK_H
#define OPS_BASE_CHECK_H

#include <variant>
#include <unordered_map>
#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorOps.h"
#include "acc/core/framework/OperatorIndex.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/ErrorCode.h"

namespace Acc {
// Range constraint type
struct RangeConstraint {
    int minVal = -1;
    int maxVal = -1;
};

// Enumerated constraint type
struct EnumeratedConstraint {
    std::vector<int> values = {};
};

// Type that can be either a range constraint or an enumerated constraint
using DimensionConstraint = std::variant<RangeConstraint, EnumeratedConstraint>;

// Constraint type for a single tensor
struct TensorConstraint {
    std::string device;
    std::vector<DataType> dataTypes;
    std::vector<TensorFormat> formats;
    std::unordered_map<std::string, DimensionConstraint> dimensionConstraints;
};

// Tensor constraints for a single operator
struct OperatorTensorConstraints {
    std::vector<TensorConstraint> inputConstraints;
    std::vector<TensorConstraint> outputConstraints;
};

class OpsBaseChecker {
public:
    /**
     * @brief Construct a new Ops Base Checker object
     *
     * @param opName the name of operator
     */
    explicit OpsBaseChecker(const OperatorId& opId);
    /**
     * @brief Destroy the Ops Base Checker object
     *
     */
    virtual ~OpsBaseChecker() = default;
    /**
     * @brief Initiates validation and implicit memory allocation
     *
     * @param checkContext context to be checked
     * @return ErrorCode
     */
    ErrorCode CheckAndImplicitMalloc(const OperatorContext& ctx);

protected:
    /**
     * @brief Iterate over each tensor to verify that individual tensors meet the constraints
     *
     * @param inputTensorRefs vector of input tensors reference
     * @param outputTensorRefs vector of output tensors reference
     * @return ErrorCode
     */
    ErrorCode CheckEachTensorValid(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                                   const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs);
    /**
     * @brief Custom validation rules; by default, checks that the properties of tensors exactly match
     *
     * @param ctx context to be checked
     * @return ErrorCode
     */
    virtual ErrorCode CheckCustomRules(const OperatorContext& ctx);
    /**
     * @brief Custom implicit memory allocation; by default, the output size is consistent with the input size.
     *
     * @param ctx context to be checked
     * @return ErrorCode
     */
    virtual ErrorCode ImplicitMalloc(const OperatorContext& ctx);
    /**
     * @brief Check whether each input/output tensor is consistent.
     *
     * @param inputTensorRefs vector of input tensors reference
     * @param outputTensorRefs vector of output tensors reference
     * @return ErrorCode
     */
    ErrorCode CheckMultiTensorMatch(const std::vector<std::reference_wrapper<const Tensor>>& inputTensorRefs,
                                    const std::vector<std::reference_wrapper<Tensor>>& outputTensorRefs);
    /**
     * @brief Check each dimension of the tensor.
     *
     * @param tensor
     * @param dimConstraints
     * @return ErrorCode
     */
    ErrorCode CheckTensorDimension(const Tensor& tensor,
                                   const std::unordered_map<std::string, DimensionConstraint>& dimConstraints) const;
    /**
     * @brief Check each attributes of the tensor.
     *
     * @param tensor
     * @param tensorConstraint
     * @return ErrorCode
     */
    ErrorCode CheckTensorAttributes(const Tensor& tensor, const TensorConstraint& tensorConstraint) const;

protected:
    OperatorId opId_;
    // Identifies whether the output is pre-assigned
    std::vector<bool> outputMallocFlags_;
};
} // namespace Acc
#endif // OPS_BASE_CHECK_H