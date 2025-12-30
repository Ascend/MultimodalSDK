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
 * Description: XPUAccelerator definition.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef XPU_ACCELERATOR_H
#define XPU_ACCELERATOR_H

#include <unordered_map>
#include <functional>
#include <iostream>

#include "OperatorContext.h"
#include "OperatorIndex.h"

#include "acc/tensor/Tensor.h"

namespace Acc {
    // Function signature for all operator implementations
    using OperatorFunc = std::function<ErrorCode(OperatorContext& opCtx)>;
    // Registry mapping operator IDs to their execution functions
    using OperatorMap = std::unordered_map<OperatorId, OperatorFunc>;

    /**
     * @brief Abstract base class for cross-platform accelerator implementations
     * @details Provides a unified interface for executing image processing operators
     *          across different hardware platforms (CPU, NPU, DVPP, etc.), current support CPU
     */
    class XPUAccelerator {
    public:
        virtual ~XPUAccelerator() = default;

        /**
         * @brief Execute a registered operator with the provided context
         * @details Looks up the operator function by ID and executes it with the given context
         * @param opId Unique identifier for the operator to execute
         * @param opCtx Context object containing all necessary data and parameters for the operation
         * @return ErrorCode
         */
        ErrorCode ExecuteOperator(OperatorId opId, OperatorContext& opCtx);

    protected:
        OperatorMap operatorMap_; // Registry of available operators for this accelerator instance

        /**
         * @brief Private template function to create operator wrappers
         *
         * @tparam ContextType The specific context type (e.g., CropContext, ResizeContext, etc.)
         * @param func Function pointer to the specific operator function that takes ContextType& parameter
         * @return OperatorFunc Returns a lambda wrapper that accepts generic OperatorContext& parameter
         */
        template<typename ContextType>
        static OperatorFunc CreateOperatorFunc(ErrorCode(*func)(ContextType&))
        {
            return [func](OperatorContext& opCtx) -> ErrorCode {
                ContextType& specificCtx = dynamic_cast<ContextType&>(opCtx);
                return func(specificCtx);
            };
        }
    };

    /**
     * @brief Factory function for getting device-specific accelerator instances
     * @param device Target device type (CPU, NPU, DVPP, etc.) for the accelerator, current support CPU
     * @return XPUAccelerator, the created accelerator instance
     */
    XPUAccelerator& GetAccelerator(DeviceMode device);
}

#endif // XPU_ACCELERATOR_H