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
 * Description: Image header file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include "acc/tensor/Tensor.h"
#include "acc/ErrorCode.h"
#include "acc/image/ImageFormat.h"

namespace Acc {

class Image {
public:
    /**
     * @brief Construct a new Image object
     *
     */
    Image() = default;
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
     * @brief Construct a new Image object from raw ptr
     *
     * @param data user input data
     * @param imSize Image size
     * @param imFormat image format, range is RGB, BGR, RGB_PLANAR, BGR_PLANAR
     * @param dataType data type, Only support UINT8
     * @param device device str, range is cpu
     */
    Image(void* data, const std::vector<size_t>& imSize, ImageFormat imFormat = ImageFormat::RGB,
          DataType dataType = DataType::UINT8, const char* device = "cpu");

    /**
     * @brief Construct a new Image object from shared_ptr
     *
     * @param data user input data
     * @param imSize Image size
     * @param imFormat image format, range is RGB, BGR, RGB_PLANAR, BGR_PLANAR
     * @param dataType data type, Only support UINT8
     * @param device device str, range is cpu
     */
    Image(std::shared_ptr<void> dataPtr, const std::vector<size_t>& imSize, ImageFormat imFormat = ImageFormat::RGB,
          DataType dataType = DataType::UINT8, const char* device = "cpu");

    /**
     * @brief Construct a new Image object from given path
     *
     * @param path user input path
     * @param device device str, range is cpu
     */
    Image(const char* path, const char* device);
    /**
     * @brief Image deep copy
     *
     * @param Image dst Image
     * @return ErrorCode
     */
    ErrorCode Clone(Image& other) const;

    /**
     * @brief Get size property
     *
     * @return const std::vector<size_t>&
     */
    const std::vector<size_t>& Size() const;
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
    ImageFormat Format() const;
    /**
     * @brief Get data type property
     *
     * @return DataType
     */
    DataType DType() const;
    /**
     * @brief Get device property
     *
     * @return std::unique_ptr<char[]>
     */
    std::unique_ptr<char[]> Device() const;
    /**
     * @brief Get raw ptr
     *
     * @return void*
     */
    void* Ptr() const;
    /**
     * @brief Get the Tensor object
     *
     * @return Tensor&
     */
    Tensor& GetTensor() const;

private:
    // mutable to modify tensor in const function
    mutable Tensor tensor_;
    ImageFormat format_ = ImageFormat::UNDEFINED;
    std::vector<size_t> size_ = {};
};
} // namespace Acc
#endif // IMAGE_H
