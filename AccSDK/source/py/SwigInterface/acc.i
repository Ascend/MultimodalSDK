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
 * Description: Python version variable definition.
 * Author: MindX SDK
 * Create: 2025
 * History: NA
 */

%module(directors="1") acc
%{
// string will be converted into python bytes
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#include "acc/tensor/TensorDataType.h"
#include "acc/image/ImageFormat.h"
#include "PyTensor.h"
#include "PyTensorOps.h"
#include "PyImage.h"
#include "PyLog.h"
#include "PyPreprocess.h"
#include "PyVideo.h"
#include "PyAudio.h"
#include "acc/Log.h"
%}
%include <std_string.i>
%include <std_vector.i>
%include <stdint.i>
%include <std_set.i>
%include <attribute.i>
%include <std_shared_ptr.i>
namespace std {
    %template(IntVector) vector<int>;
    %template(StringVector) std::vector<string>;
    %template(SizetVector) vector<size_t>;
    %template(FloatVector) std::vector<float>;
    %template(Tensorvector) std::vector<PyAcc::Tensor>;
    %template(Imagevector) std::vector<PyAcc::Image>;
    %template(Uint32_tSet) set<uint32_t>;
}
%exception {
    try {
        $action;
    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        SWIG_fail;
    }
}
%feature("director:except") {
if ($error != NULL) {
throw Swig::DirectorMethodException();
}
}

// Tensor wrapper
%attribute(PyAcc::Tensor, std::string, device, PyAcc::Tensor::Device);
%attribute(PyAcc::Tensor, Acc::DataType, dtype, PyAcc::Tensor::Dtype);
%attributeval(PyAcc::Tensor, std::vector<size_t>, shape, PyAcc::Tensor::Shape);
%attribute(PyAcc::Tensor, Acc::TensorFormat, format, PyAcc::Tensor::Format);
%attribute(PyAcc::Tensor, size_t, nbytes, PyAcc::Tensor::NumBytes);
%ignore PyAcc::Tensor::Device();
%ignore PyAcc::Tensor::Shape() const;
%ignore PyAcc::Tensor::Dtype() const;
%ignore PyAcc::Tensor::Format() const;
%ignore PyAcc::Tensor::NumBytes() const;
%ignore PyAcc::Tensor::SetTensor(const Acc::Tensor& src);
%ignore PyAcc::Tensor::GetTensorPtr() const;
%ignore Acc::TensorAuxInfo;

// Image wrapper
%attribute(PyAcc::Image, std::string, device, PyAcc::Image::Device);
%attribute(PyAcc::Image, Acc::DataType, dtype, PyAcc::Image::Dtype);
%attributeval(PyAcc::Image, std::vector<size_t>, size, PyAcc::Image::Size);
%attribute(PyAcc::Image, Acc::ImageFormat, format, PyAcc::Image::Format);
%attribute(PyAcc::Image, size_t, nbytes, PyAcc::Image::NumBytes);
%attribute(PyAcc::Image, size_t, height, PyAcc::Image::Height);
%attribute(PyAcc::Image, size_t, width, PyAcc::Image::Width);
%ignore PyAcc::Image::Image();
%ignore PyAcc::Image::Height() const;
%ignore PyAcc::Image::Width() const;
%ignore PyAcc::Image::Device() const;
%ignore PyAcc::Image::Size() const;
%ignore PyAcc::Image::Dtype() const;
%ignore PyAcc::Image::Format() const;
%ignore PyAcc::Image::NumBytes() const;
%ignore PyAcc::Image::GetImagePtr() const;
%ignore PyAcc::Image::setImage(const Acc::Image& src);

// Log wrapper
%ignore Acc::RegisterLogFn(LogFn fn, LogLevel minLevel);
%feature("director") PyAcc::LogCallBacker;
%include "PyTensor.h"
%include "PyTensorOps.h"
%include "PyImage.h"
%include "PyLog.h"
%include "PyPreprocess.h"
%include "acc/tensor/TensorDataType.h"
%include "PyVideo.h"
%include "PyAudio.h"
%include "acc/image/ImageFormat.h"
%include "acc/Log.h"