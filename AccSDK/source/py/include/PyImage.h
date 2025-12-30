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
#ifndef PYIMAGE_H
#define PYIMAGE_H
#include <vector>
#include <memory>
#include "Python.h"

#include "PyTensor.h"
#include "acc/tensor/Tensor.h"
#include "acc/image/Image.h"
#include "acc/image/ImageFormat.h"
namespace PyAcc {
class Image {
public:
    /**
     * @brief Destroy a new Image object
     *
     */
    ~Image() = default;
    /**
     * @brief Construct a new Image object from copy
     *
     * @param other Input Image
     */
    Image(const Image& other) = default;
    /**
     * @brief assignment operation
     *
     * @param other Input Image
     * @return Image& self Image
     */
    Image& operator=(const Image& other) = default;
    /**
     * @brief Construct a new Image object from given path
     *
     * @param path user input path
     * @param device device str, range is cpu
     */
    Image(const char* path, const char* device);
    /**
     * @brief Image deep copy, exposed for Python
     *
     * @param Image dst Image
     * @return Image
     */
    Image clone() const;
    /**
     * @brief Convert Image object to numpy array, exposed to Python
     *
     * @return PyObject which will be converted into numpy array
     */
    PyObject* numpy();

    // inner aux funtion
    /**
     * @brief Construct a new Image object
     *
     */
    Image();
    /**
     * @brief Get Image shared ptr
     *
     * @return std::shared_ptr<Acc::Image>
     */
    std::shared_ptr<Acc::Image> GetImagePtr() const;
    /**
     * @brief Get size property
     *
     * @return const std::vector<size_t>&
     */
    std::vector<size_t> Size() const;
    /**
     * @brief Get Width property
     *
     * @return size_t
     */
    size_t Width() const;
    /**
     * @brief Get Height property
     *
     * @return size_t
     */
    size_t Height() const;
    /**
     * @brief Get num bytes property
     *
     * @return size_t
     */
    size_t NumBytes() const;
    /**
     * @brief Get ImageFormat property
     *
     * @return ImageFormat
     */
    Acc::ImageFormat Format() const;
    /**
     * @brief Get data type property
     *
     * @return DataType
     */
    Acc::DataType Dtype() const;
    /**
     * @brief Get device property
     *
     * @return const std::string& Returns a string to simplify swig wrap and conversion to a
     * Python built-in type.
     */
    const std::string& Device();
    /**
     * @brief set internal image ptr
     *
     * @param src: source Image
     */
    void SetImage(const Acc::Image& src);
    /**
     * @brief Construct Image from given path, exposed for Python
     *
     * @param Image dst Image
     * @return Image
     */
    static Image open(const std::string& path, const std::string& device);
    /**
     * @brief Construct Image from numpy array, exposed for Python
     *
     * @param PyObject: numpy array which is a PyObject in C++
     * @param ImageFormat: Image format
     * @param device device str, range is cpu
     * @return Image
     */
    static Image from_numpy(PyObject* pyObj, Acc::ImageFormat imageFormat, const char* device);
    /**
     * @brief Image resize
     *
     * @param resize_w resized width
     * @param resize_h resized height
     * @param interpolation interpolation algorithm
     * @param device_mode the mode for running operator
     * @return Image new image
     */
    Image resize(size_t resize_w, size_t resize_h, Acc::Interpolation interpolation = Acc::Interpolation::BICUBIC,
                 Acc::DeviceMode device_mode = Acc::DeviceMode::CPU);

    /**
     * @brief Image crop
     *
     * @param top: top boundary position of the crop
     * @param left: left boundary position of the crop
     * @param height: crop height
     * @param width: crop width
     * @param device_mode the mode for running operator
     * @return Image
     */
    Image crop(uint32_t top, uint32_t left, uint32_t height, uint32_t width,
               Acc::DeviceMode device_mode = Acc::DeviceMode::CPU);

    /**
     * @brief Image to_tensor
     *
     * @param format target format, support NHWC、NCHW
     * @param device_mode the mode for running operator
     * @return
     */
    Tensor to_tensor(Acc::TensorFormat format, Acc::DeviceMode device_mode = Acc::DeviceMode::CPU);

private:
    std::shared_ptr<Acc::Image> image_ = nullptr;
    std::string deviceStr_ = "cpu";
};

} // namespace PyAcc

#endif // PYIMAGE_H
