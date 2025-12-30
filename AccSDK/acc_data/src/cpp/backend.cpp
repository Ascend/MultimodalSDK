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
 * @Date: 2025-2-14 11:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-14 11:00:00
 */
#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "accdata_pipeline.h"
#include "accdata_tensor.h"
#include "logger.h"
#include "accdata_op_spec.h"
#include "accdata_error_code.h"

namespace acclib {
namespace accdata {

namespace py = pybind11;
using namespace pybind11::literals;

AccDataErrorCode IsContiguousBuffer(ssize_t itemSize, const std::vector<ssize_t>& shape,
    const std::vector<ssize_t>& strides)
{
    if (shape.size() == 0 || shape.size() != strides.size()) {
        return AccDataErrorCode::H_TENSOR_ERROR;
    }
    for (int i = static_cast<int>(strides.size()) - 1; i >= 0; --i) {
        if (strides[i] != itemSize) {
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        itemSize *= shape[i];
    }
    return AccDataErrorCode::H_OK;
}

TensorDataType ParseDataType(const std::string& format)
{
    char letter;
    int number = -1;
    uint64_t minFormatSize = 1;
    uint64_t maxFormatSize = 2;

    if (format.size() == minFormatSize) {
        letter = format[minFormatSize - 1];
    } else if (format.size() > minFormatSize) {
        letter = format[maxFormatSize - 1];
        if (format.size() > maxFormatSize) {
            try {
                number = std::stoi(format.substr(maxFormatSize));
            } catch(...) {
                letter = '?';
            }
        }
    } else {
        letter = '?';
    }
    switch (letter) {
        case 'B':
            return TensorDataType::UINT8;
        case 'f':
            if (number == -1 || number == sizeof(float)) {
                return TensorDataType::FP32;
            }
            break;
        default:
            break;
    }
    return TensorDataType::LAST;
}

std::string EncodeDataType(TensorDataType type)
{
    switch (type) {
        case TensorDataType::UINT8:
            return "=b";
        case TensorDataType::FP32:
            return "=f";
        default:
            break;
    }
    return "=?";
}

template <bool SHARE>
static AccDataErrorCode CopyExternalBuffer(AccDataTensor& inTensor, const py::buffer& b)
{
    auto errCode = AccDataErrorCode::H_OK;
    py::buffer_info info = b.request();

    errCode = IsContiguousBuffer(info.itemsize, info.shape, info.strides);
    if (errCode != AccDataErrorCode::H_OK) {
        return errCode;
    }

    TensorShape shape(info.shape.size());
    for (uint64_t i = 0; i < info.shape.size(); ++i) {
        shape[i] = static_cast<size_t>(info.shape[i]);
    }

    auto dataType = ParseDataType(info.format);
    if constexpr (SHARE) {
        /* Keep a copy of the input buffer in the deleter, so its refcount is increased. When only
            C++ holds the reference, we need to get the GIL before we free the object, so use dynamically
            allocated memory. */
        py::buffer* tmp = new (std::nothrow) py::buffer(b);
        if (tmp == nullptr) {
            return AccDataErrorCode::H_TENSOR_ERROR;
        }
        auto deleter = [ref = tmp](void*) {
            py::gil_scoped_acquire aqr;
            delete ref;
        };
        return inTensor.ShareData(std::shared_ptr<void>(info.ptr, deleter), shape, dataType);
    } else {
        return inTensor.Copy(static_cast<void*>(info.ptr), shape, dataType);
    }
}

template <bool SHARE>
static AccDataErrorCode CopyExternalStr(AccDataTensor& inTensor, const py::str& s)
{
    std::string ss = s.cast<std::string>();
    TensorShape shape({ss.size()});

    /* Assume that the string is not large, always copy. */
    auto errCode = inTensor.Copy(static_cast<void*>(ss.data()), shape, TensorDataType::CHAR);
    if (errCode != AccDataErrorCode::H_OK) {
        return AccDataErrorCode::H_TENSOR_ERROR;
    }

    return AccDataErrorCode::H_OK;
}

template <bool SHARE>
static void CopyExternalData(AccDataTensor& inTensor, const py::object& object,
                             TensorLayout layout = TensorLayout::PLAIN)
{
    std::ostringstream oss;
    if (py::isinstance<py::buffer>(object)) {
        auto errCode = CopyExternalBuffer<SHARE>(inTensor, object.cast<py::buffer>());
        if (errCode != AccDataErrorCode::H_OK) {
            oss << "copy external buffer failed, errCode is: " << errCode;
            throw std::runtime_error(oss.str());
        }
    } else if (py::isinstance<py::str>(object)) {
        auto errCode = CopyExternalStr<SHARE>(inTensor, object.cast<py::str>());
        if (errCode != AccDataErrorCode::H_OK) {
            oss << "copy external str failed, errCode is: " << errCode;
            throw std::runtime_error(oss.str());
        }
    } else {
        throw std::runtime_error("Unsupported python type.");
    }
    inTensor.SetLayout(layout);
}

static AccDataErrorCode ShareTorchData(AccDataTensor& inTensor, const py::buffer& b, const std::vector<size_t>& tShape,
    TensorLayout layout, TensorDataType dtype)
{
    py::buffer_info info = b.request();
    py::buffer* tmp = new (std::nothrow) py::buffer(b);
    if (tmp == nullptr) {
        return AccDataErrorCode::H_TENSOR_ERROR;
    }
    auto deleter = [ref = tmp](void*) {
        py::gil_scoped_acquire aqr;
        delete ref;
    };

    inTensor.SetLayout(layout);
    return inTensor.ShareData(std::shared_ptr<void>(info.ptr, deleter), tShape, dtype);
}

static std::string GetDataNodeName(const py::object& object)
{
    if (py::isinstance<py::str>(object)) {
        return object.cast<std::string>();
    } else if (py::hasattr(object, "name")) {
        return object.attr("name").cast<std::string>();
    } else {
        throw std::runtime_error("Expected a string or an object with a 'name' attribute");
    }
}

static void ExposeOpSpec(py::module& m)
{
#define ADD_ARG(T)                                                                  \
    [](AccDataOpSpec* spec, const std::string& name, T value, bool overwrite)->AccDataOpSpec& {   \
        spec->AddArg(name, value, overwrite);                                       \
        return *spec;                                                               \
    },                                                                              \
        "name"_a, "value"_a, "overwrite"_a = true

#define DEF_ADD_ARG(T) .def("AddArg", ADD_ARG(T), policy).def("AddArg", ADD_ARG(std::vector<T>), policy)

    constexpr auto policy = py::return_value_policy::reference_internal;
    py::class_<AccDataOpSpec, std::shared_ptr<AccDataOpSpec>>(m, "OpSpec")
        .def("AddInput", &AccDataOpSpec::AddInput, "name"_a, "device"_a = "cpu", policy)
        .def("AddOutput", &AccDataOpSpec::AddOutput, "name"_a, "device"_a = "cpu", policy)
            DEF_ADD_ARG(bool) DEF_ADD_ARG(int64_t) DEF_ADD_ARG(float) DEF_ADD_ARG(std::string);
    m.def("new_op", &AccDataOpSpec::Create, "name"_a, "Create a new AccDataPipeline instance");

#undef DEF_ADD_ARG
#undef ADD_ARG
}

static void ExposePipeline(py::module& m)
{
    auto run = [](AccDataPipeline& p, std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> inputs,
        bool copy) {
        std::vector<std::shared_ptr<AccDataTensorList>> opOutputs;
        auto errCode = p.Run(inputs, opOutputs, copy);
        if (errCode != AccDataErrorCode::H_OK) {
            throw std::runtime_error("Pipeline run failed with error code: " +
                std::to_string(static_cast<int>(errCode)));
            return py::tuple();
        }
        py::tuple outs(opOutputs.size());
        for (uint64_t i = 0 ; i < opOutputs.size() ; ++i) {
            outs[i] = opOutputs[i];
        }
        return outs;
    };

    auto build = [](AccDataPipeline* pipe, const std::vector<std::shared_ptr<AccDataOpSpec>>& specs,
        const std::vector<py::object>& outputs) {
        std::vector<std::string> outputVec;
        for (auto &elm : outputs) {
            outputVec.push_back(GetDataNodeName(elm));
        }
        return pipe->Build(specs, outputVec);
    };

    py::class_<AccDataPipeline, std::shared_ptr<AccDataPipeline>>(m, "Pipeline")
        .def("Build", build)
        .def("Run", run, py::return_value_policy::take_ownership);
    m.def("new_instance", &AccDataPipeline::Create, "Create a new AccDataPipeline instance");
}

static void ExposeTypes(py::module& m)
{
    py::module typesModule = m.def_submodule("types");
    typesModule.doc() = "Datatypes and options used by AccData";

    // TensorDataType
    py::enum_<TensorDataType>(typesModule, "TensorDataType")
        .value("UINT8", TensorDataType::UINT8)
        .value("FP32", TensorDataType::FP32)
        .value("CHAR", TensorDataType::CHAR)
        .export_values();

    // TensorLayout
    py::enum_<TensorLayout>(typesModule, "TensorLayout")
        .value("PLAIN", TensorLayout::PLAIN)
        .value("NHWC", TensorLayout::NHWC)
        .value("NCHW", TensorLayout::NCHW)
        .export_values();
}

py::list PyTensorShape(const AccDataTensor& t)
{
    py::list shape;
    auto tShape = t.Shape();

    for (uint64_t i = 0; i < tShape.size(); i++) {
        shape.append(tShape[i]);
    }
    return shape;
}

static void ExposeTensor(py::module& m)
{
    auto rawDataPtr = [](AccDataTensor& inTensor) {
        return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(inTensor.RawDataPtr().get()));
    };

    auto pybuffer = [](AccDataTensor& inTensor) -> py::buffer_info {
        auto& shape = inTensor.Shape();
        uint64_t numDims = shape.size();
        int typeSize = TensorDataTypeSize(inTensor.DataType());
        std::vector<ssize_t> tmpShape(numDims);
        std::vector<ssize_t> tmpStride(numDims);
        int stride = 1;
        for (uint64_t i = 0; i < numDims; ++i) {
            tmpShape[i] = shape[i];
            tmpStride[numDims - 1 - i] = typeSize * stride;
            stride *= shape[numDims - 1 - i];
        }
        return py::buffer_info(inTensor.RawDataPtr().get(), typeSize, EncodeDataType(inTensor.DataType()),
                               numDims, tmpShape, tmpStride);
    };

    py::class_<AccDataTensor, std::shared_ptr<AccDataTensor>>(m, "Tensor", py::buffer_protocol())
        .def_buffer(pybuffer)
        .def("Copy", CopyExternalData<false>, "b"_a, "layout"_a = TensorLayout::PLAIN)
        .def("ShareData", CopyExternalData<true>, "b"_a, "layout"_a = TensorLayout::PLAIN)
        .def("ShareTorchData", ShareTorchData, "ptr"_a, "shape"_a, "layout"_a, "dtype"_a)
        .def("RawDataPtr", rawDataPtr)
        .def("Layout", &AccDataTensor::Layout)
        .def("DataType", &AccDataTensor::DataType)
        .def("Shape", &PyTensorShape);
}

static void ExposeTensorList(py::module& m)
{
    constexpr auto policy = py::return_value_policy::reference_internal;
    py::class_<AccDataTensorList, std::shared_ptr<AccDataTensorList>>(m, "TensorList")
        .def("__len__", &AccDataTensorList::NumTensors)
        .def("__getitem__", [](AccDataTensorList& list, int idx) -> AccDataTensor& { return list[idx]; }, policy);
    m.def("new_tensorlist", &AccDataTensorList::Create, "batchSize"_a, "Create a new AccDataTensorList instance");
}

static void ExposeErrorCode(py::module& m)
{
    py::enum_<AccDataErrorCode>(m, "ErrorCode")
        .value("H_OK", AccDataErrorCode::H_OK, "H_OK")
        .value("H_COMMON_ERROR", AccDataErrorCode::H_COMMON_ERROR, "H_COMMON_ERROR")
        .value("H_COMMON_UNKNOWN_ERROR", AccDataErrorCode::H_COMMON_UNKNOWN_ERROR, "H_COMMON_UNKNOWN_ERROR")
        .value("H_COMMON_LOGGER_ERROR", AccDataErrorCode::H_COMMON_LOGGER_ERROR, "H_COMMON_LOGGER_ERROR")
        .value("H_COMMON_INVALID_PARAM", AccDataErrorCode::H_COMMON_INVALID_PARAM, "H_COMMON_INVALID_PARAM")
        .value("H_COMMON_OPERATOR_ERROR", AccDataErrorCode::H_COMMON_OPERATOR_ERROR, "H_COMMON_OPERATOR_ERROR")
        .value("H_COMMON_NULLPTR", AccDataErrorCode::H_COMMON_NULLPTR, "H_COMMON_NULLPTR")
        .value("H_SINGLEOP_ERROR", AccDataErrorCode::H_SINGLEOP_ERROR, "H_SINGLEOP_ERROR")
        .value("H_FUSIONOP_ERROR", AccDataErrorCode::H_FUSIONOP_ERROR, "H_FUSIONOP_ERROR")
        .value("H_USEROP_ERROR", AccDataErrorCode::H_USEROP_ERROR, "H_USEROP_ERROR")
        .value("H_PIPELINE_ERROR", AccDataErrorCode::H_PIPELINE_ERROR, "H_PIPELINE_ERROR")
        .value("H_PIPELINE_BUILD_ERROR", AccDataErrorCode::H_PIPELINE_BUILD_ERROR, "H_PIPELINE_BUILD_ERROR")
        .value("H_PIPELINE_STATE_ERROR", AccDataErrorCode::H_PIPELINE_STATE_ERROR, "H_PIPELINE_STATE_ERROR")
        .value("H_TENSOR_ERROR", AccDataErrorCode::H_TENSOR_ERROR, "H_TENSOR_ERROR")
        .value("H_THREADPOOL_ERROR", AccDataErrorCode::H_THREADPOOL_ERROR, "H_THREADPOOL_ERROR")
        .export_values();
}

PYBIND11_MODULE(backend, m)
{
    m.doc() = R"pbdoc(Python bindings for the C++ layer of acc_data)pbdoc";
    m.def("SetLogLevel", &Logger::SetLogLevelStr, "level"_a = "info",
          "Set log level for the underlying C++ layer. Optional levels are debug|info|warn|error");

    ExposeTypes(m);
    ExposeOpSpec(m);
    ExposePipeline(m);
    ExposeTensor(m);
    ExposeTensorList(m);
    ExposeErrorCode(m);
}

} // namespace accdata
} // namespace acclib
