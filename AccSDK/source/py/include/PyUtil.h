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
 * Description: tensor file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef PYUTIL_H
#define PYUTIL_H
#include <vector>
#include "Python.h"
#include "acc/tensor/TensorDataType.h"

namespace PyAcc {
    /**
     * @brief Construct the data required for numpy's __array_interface__
     *
     * @param dataPtr data pointer, used to point to the address where the data exists
     * @param shape numpy array shape, it will be used for tensor construction
     * @param dataType numpy array data type, it will be used for tensor construction and must be [int8/uint8/float32]
     */
    struct NumpyData {
        void* dataPtr;
        std::vector<size_t> shape;
        Acc::DataType dataType;
    };

    /**
     * @brief Get NumpyData from python numpy object
     * @param pyObj python numpy object
     * @return NumpyData the value must has data ptr, shape and data type
     */
    NumpyData GetNumpyData(PyObject* pyObj);

    /**
     * @brief Support conversion NumpyData to numpy ndarray
     * @param numpyData python numpy ndarray
     * @return python numpy ndarray with __array_interface__ dict
     */
    PyObject* ToNumpy(NumpyData numpyData);
}

#endif // PYUTIL_H
