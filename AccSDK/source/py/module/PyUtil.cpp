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
#include "PyUtil.h"

#include <vector>
#include <iostream>
#include <map>

#include "PyTensor.h"
#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorDataType.h"

namespace {
    long ARRAY_INTERFACE_VERSION = 3;
    const size_t DATA_FIELD_ELEMENTS = 2;
    const std::map<std::string, Acc::DataType> FORMAT_TO_DATA_TYPE_MAP = {
        {"<i1", Acc::DataType::INT8}, {"|i1", Acc::DataType::INT8}, {">i1", Acc::DataType::INT8},
        {"<u1", Acc::DataType::UINT8}, {"|u1", Acc::DataType::UINT8}, {">u1", Acc::DataType::UINT8},
        {"<f4", Acc::DataType::FLOAT32}, {"|f4", Acc::DataType::FLOAT32}, {">f4", Acc::DataType::FLOAT32}
    };
    const std::map<Acc::DataType, std::string> DATA_TYPE_TO_FORMAT = {
        {Acc::DataType::INT8, "|i1"},
        {Acc::DataType::UINT8, "|u1"},
        {Acc::DataType::FLOAT32, "<f4"} // Little-endian
    };
}
namespace PyAcc {
    NumpyData GetNumpyData(PyObject* pyObj)
    {
        if (!pyObj) {
            throw std::runtime_error("The python numpy ndarray is empty, please ensure the input data is correct");
        }

        // The __array_interface__ protocol defines a method for accessing NumPy array contents from other C extensions.
        // It only requires obtaining the data pointer, shape, and data type.
        PyObject* arrayInterface = PyObject_GetAttrString(pyObj, "__array_interface__");
        if (!arrayInterface) {
            throw std::runtime_error("The python numpy ndarray does not have the __array_interface__ dictionary in "
                                     "its attributes. Please check whether the passed numpy ndarray is corrupted.");
        }

        // Wrap the PyObject* in a smart pointer for automatic reference count management
        // The custom deleter ensures Py_DECREF is called when the shared_ptr goes out of scope
        // This provides RAII (Resource Acquisition Is Initialization) for Python object references
        std::shared_ptr<PyObject> arrayInterfacePtr(arrayInterface, [](PyObject* obj) {
           if (obj != nullptr) {
               // Decrease reference count when smart pointer is destroyed
               Py_DECREF(obj);
           }
        });

        NumpyData numpyData;
        // The data pointer exists in the "data" field of __array_interface__. It must contain a 2-tuple.
        // The first element is data address. The second element is a read-only flag (true means read-only).
        // Borrowed reference, no new object created, no memory management required.
        PyObject *dataTuple = PyDict_GetItemString(arrayInterface, "data");
        if (dataTuple == nullptr || PyTuple_Check(dataTuple) != 1 || PyTuple_Size(dataTuple) != DATA_FIELD_ELEMENTS) {
            throw std::runtime_error("Invalid data field in __array_interface__ of python numpy ndarray, "
                                     "It shouldn't be missing and contain a 2-tuple (address, read-only flag).");
        }

        // Get the first element (data pointer) from the data tuple
        // PyTuple_GetItem returns a borrowed reference, so no need to call Py_DECREF
        PyObject *dataPtrObj = PyTuple_GetItem(dataTuple, 0);
        numpyData.dataPtr = reinterpret_cast<void*>(PyLong_AsVoidPtr(dataPtrObj));
        if (PyErr_Occurred() || !numpyData.dataPtr) {
            throw std::runtime_error("Failed to get valid data pointer from __array_interface__ of python numpy "
                                     "ndarray. The data field's address must be legal");
        }

        // The shape exists in the "shape" field of __array_interface__.
        // Borrowed reference, no new object created, no memory management required.
        PyObject *shapeTuple = PyDict_GetItemString(arrayInterface, "shape");
        if (shapeTuple == nullptr || PyTuple_Check(shapeTuple) != 1 || PyTuple_Size(shapeTuple) == 0) {
            throw std::runtime_error("Invalid shape field in __array_interface__ of python numpy ndarray. "
                                     "It shouldn't be missing and shouldn't be empty tuple.");
        }
        for (Py_ssize_t i = 0; i < PyTuple_Size(shapeTuple); i++) {
            PyObject *dim = PyTuple_GetItem(shapeTuple, i);
            size_t dimSize = PyLong_AsSize_t(dim);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Invalid dimension in shape of __array_interface__ of python numpy "
                                         "ndarray. The dimension's value must greater than 0.");
            }
            numpyData.shape.push_back(dimSize);
        }

        // The data type exists in the "typestr" field of __array_interface__.
        // The basic string format consists of 3 parts: a character describing the byteorder of the data
        // <: little-endian, >: big-endian, |: not-relevant
        PyObject *typeStrObj = PyDict_GetItemString(arrayInterface, "typestr");
        if (typeStrObj == nullptr || PyUnicode_Check(typeStrObj) != 1) {
            throw std::runtime_error("Invalid typestr field in __array_interface__ of python numpy ndarray. "
                                     "It shouldn't be missing and must be a Unicode object.");
        }
        // Creates new object, requires reference count management.
        PyObject *typeStrBytes = PyUnicode_AsEncodedString(typeStrObj, "utf-8", "strict");
        if (!typeStrBytes) {
            throw std::runtime_error("Encode the typestr field's value in __array_interface__ to string failed. "
                                     "Python interpreter may be out of memory.");
        }
        const char *typeStrCstr = PyBytes_AsString(typeStrBytes);

        // Create C++ string copy from the C string
        // This creates an independent copy, so we can safely release the Python object
        std::string typeStr(typeStrCstr);
        Py_DECREF(typeStrBytes); // reference count management

        auto it = FORMAT_TO_DATA_TYPE_MAP.find(typeStr);
        if (it != FORMAT_TO_DATA_TYPE_MAP.end()) {
            numpyData.dataType = it->second;
        } else {
            throw std::runtime_error("Unsupported python numpy ndarray datatype. "
                                     "Only support int8, uint8, float32.");
        }

        return numpyData;
    }

    PyObject* ToNumpy(NumpyData numpyData)
    {
        // When converting to numpy, it must be ensured that the tensor instance can be created,
        // therefore there is no need to validate the legality of numpyData.

        /// Create __array_interface__ dictionary
        PyObject* interface = PyDict_New();
        if (!interface) {
            throw std::runtime_error("Unexpected error, failed to create interface dictionary for __array_interface__ "
                                     "dictionary. Python interpreter may be out of memory.");
        }

        // Set version field (always 3 for current numpy array interface)
        PyDict_SetItemString(interface, "version", PyLong_FromLong(ARRAY_INTERFACE_VERSION));

        // Create data tuple containing (data address, read_only_flag)
        PyObject* data_tuple = PyTuple_New(DATA_FIELD_ELEMENTS);
        PyTuple_SetItem(data_tuple, 0, PyLong_FromVoidPtr(numpyData.dataPtr));

        // Get reference to the global False singleton(Py_False) object (borrowed reference)
        PyObject* read_only_flag = Py_False;
        // Increment reference count before passing to PyTuple_SetItem
        // This is necessary because PyTuple_SetItem "steals" the reference
        // Without Py_INCREF, the global Py_False singleton's refcount would be incorrectly decremented
        Py_INCREF(read_only_flag);
        PyTuple_SetItem(data_tuple, 1, read_only_flag);
        PyDict_SetItemString(interface, "data", data_tuple);
        Py_DECREF(data_tuple);

        // Create shape tuple from shape vector
        PyObject* shape_tuple = PyTuple_New(numpyData.shape.size());
        for (size_t i = 0; i < numpyData.shape.size(); i++) {
            PyTuple_SetItem(shape_tuple, i, PyLong_FromSize_t(numpyData.shape[i]));
        }
        PyDict_SetItemString(interface, "shape", shape_tuple);
        Py_DECREF(shape_tuple);

        // Set typestr (data type description string)
        PyDict_SetItemString(interface, "typestr",
                             PyUnicode_FromString(DATA_TYPE_TO_FORMAT.find(numpyData.dataType)->second.c_str()));

        // Create the final object containing __array_interface__
        PyObject* obj = PyDict_New();
        if (!obj) {
            Py_DECREF(interface);
            throw std::runtime_error("Failed to set __array_interface__ attribute on return object."
                                     " Python interpreter may be out of memory.");
        }

        PyDict_SetItemString(obj, "__array_interface__", interface);
        Py_DECREF(interface);
        return obj;
    }
}