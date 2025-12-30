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
#include "PyImage.h"
#include <iostream>
#include <string>

#include "PyUtil.h"
#include "PyTensor.h"
#include "acc/image/Image.h"
#include "acc/image/ImageOps.h"
#include "acc/tensor/TensorOps.h"
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"

namespace {
constexpr size_t TWO = 2;
constexpr size_t THREE = 3;
} // namespace

namespace PyAcc {

Image::Image()
{
    try {
        image_ = std::make_shared<Acc::Image>();
    } catch (const std::exception& ex) {
        image_ = nullptr;
    }
    if (image_ == nullptr) {
        Acc::LogError << "Create Image object failed. Failed to allocate memory.";
        throw std::runtime_error("Create Image object failed. Failed to allocate memory.");
    }
}

Image::Image(const char* path, const char* device)
{
    try {
        image_ = std::make_shared<Acc::Image>(path, device);
    } catch (const std::exception& ex) {
        image_ = nullptr;
    }
    if (image_ == nullptr) {
        Acc::LogError << "Create Image object failed. Failed to allocate memory.";
        throw std::runtime_error("Create Image object failed. Failed to allocate memory.");
    }
}

Image Image::clone() const
{
    Image image;
    Acc::ErrorCode ret = image_->Clone(*image.GetImagePtr());
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error("Image clone failed.");
    }
    return image;
}

std::vector<size_t> Image::Size() const
{
    return image_->Size();
}

size_t Image::Width() const
{
    return image_->Width();
}

size_t Image::Height() const
{
    return image_->Height();
}

size_t Image::NumBytes() const
{
    return image_->NumBytes();
}

Acc::ImageFormat Image::Format() const
{
    return image_->Format();
}

Acc::DataType Image::Dtype() const
{
    return image_->DType();
}

const std::string& Image::Device()
{
    std::unique_ptr<char[]> device = image_->Device();
    std::string tmpStr(device.get());
    deviceStr_ = tmpStr;
    return deviceStr_;
}

std::shared_ptr<Acc::Image> Image::GetImagePtr() const
{
    return image_;
}

void Image::SetImage(const Acc::Image& src)
{
    (*image_) = src;
}

PyObject* Image::numpy()
{
    NumpyData numpyData;
    numpyData.dataType = image_->DType();
    numpyData.dataPtr = image_->Ptr();

    auto shape = image_->Size();
    auto fmt = image_->Format();
    if (fmt == Acc::ImageFormat::RGB || fmt == Acc::ImageFormat::BGR) {
        // [H, W, 3]
        if (shape.size() == TWO) {
            std::swap(shape[0], shape[1]);
            shape.push_back(THREE);
        }
    } else if (fmt == Acc::ImageFormat::RGB_PLANAR || fmt == Acc::ImageFormat::BGR_PLANAR) {
        // [3, H, W]
        if (shape.size() == TWO) {
            std::swap(shape[0], shape[1]);
            shape.insert(shape.begin(), THREE);
        }
    } else {
        throw std::runtime_error("Unsupported image format for numpy()");
    }

    numpyData.shape = std::move(shape);
    return ToNumpy(numpyData);
}

Image Image::open(const std::string& path, const std::string& device)
{
    return PyAcc::Image(path.c_str(), device.c_str());
}

Image Image::from_numpy(PyObject* pyObj, Acc::ImageFormat imageFormat, const char* device)
{
    NumpyData numpyData = GetNumpyData(pyObj);
    if (numpyData.shape.size() != THREE) {
        throw std::runtime_error("Create Image from numpy array failed, shape should be 3D");
    }

    if (numpyData.dataType != Acc::DataType::UINT8) {
        throw std::runtime_error("Create Image from numpy array failed, data type should be uint8");
    }
    const auto& numpyShape = numpyData.shape;
    std::vector<size_t> imSize;

    switch (imageFormat) {
        case Acc::ImageFormat::RGB:
            [[fallthrough]];
        case Acc::ImageFormat::BGR:
            if (numpyShape[TWO] != THREE) {
                throw std::runtime_error(
                    std::string(
                        "Create Image from numpy array failed: for RGB/BGR expect shape [H, W, 3], got channel = ") +
                    std::to_string(numpyShape[TWO]));
            }
            imSize = {numpyShape[1], numpyShape[0]};
            break;

        case Acc::ImageFormat::RGB_PLANAR:
            [[fallthrough]];
        case Acc::ImageFormat::BGR_PLANAR:
            if (numpyShape[0] != THREE) {
                throw std::runtime_error(std::string("Create Image from numpy array failed: for RGB_PLANAR/BGR_PLANAR "
                                                     "expect shape [3, H, W], got channel = ") +
                                         std::to_string(numpyShape[0]));
            }
            imSize = {numpyShape[2], numpyShape[1]};
            break;

        default:
            throw std::runtime_error("Create Image from numpy array failed: unsupported image format");
    }

    Acc::Image imgAcc(numpyData.dataPtr, imSize, imageFormat, numpyData.dataType, device);
    Image img;
    img.SetImage(imgAcc);
    return img;
}

Image Image::resize(size_t resize_w, size_t resize_h, Acc::Interpolation interpolation, Acc::DeviceMode device_mode)
{
    Acc::Image dst;
    Acc::ErrorCode ret = Acc::ImageResize(*image_, dst, resize_w, resize_h, interpolation, device_mode);
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error("Image resize failed. Please check the detailed log above for the cause.");
    }
    Image img;
    img.SetImage(dst);
    return img;
}

Image Image::crop(uint32_t top, uint32_t left, uint32_t height, uint32_t width, Acc::DeviceMode device_mode)
{
    Acc::Image dst;
    Acc::ErrorCode ret = Acc::ImageCrop(*image_, dst, top, left, height, width, device_mode);
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error("Image crop failed. Please check the detailed log above for the cause.");
    }
    Image img;
    img.SetImage(dst);
    return img;
}

Tensor Image::to_tensor(Acc::TensorFormat format, Acc::DeviceMode device_mode)
{
    Acc::Tensor srcAccTensor = image_->GetTensor();
    Acc::Tensor outputAccTensor;
    Acc::ErrorCode ret = Acc::TensorToTensor(srcAccTensor, outputAccTensor, format, device_mode);
    if (ret != Acc::SUCCESS) {
        throw std::runtime_error("Failed to execute 'to tensor' operator, please ensure your inputs are valid.");
    }

    Tensor outputPyTensor;
    outputPyTensor.SetTensor(outputAccTensor);
    return outputPyTensor;
}
} // namespace PyAcc