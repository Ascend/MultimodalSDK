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
 * Description: Image utils file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include <string>
#include <cstdio>
#include <turbojpeg.h>
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/FileUtils.h"
#include "acc/utils/ErrorCodeUtils.h"
#include "acc/utils/ImageUtils.h"

namespace {
constexpr size_t THREE_CHANNEL = 3;
constexpr size_t MAX_WIDTH = 8192;
constexpr size_t MIN_WIDTH = 10;
constexpr size_t MAX_HEIGHT = 8192;
constexpr size_t MIN_HEIGHT = 10;
constexpr size_t IMAGE_SIZE_DIMS = 2;
constexpr size_t IMAGE_MAX_FILE_SIZE = 1024 * 1024 * 50; // 1GB
} // namespace

namespace Acc {
ErrorCode CheckImSize(const std::vector<size_t>& imSize)
{
    if (imSize.size() != IMAGE_SIZE_DIMS) {
        LogError << "Image size must be a vector of length 2." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (imSize[0] < MIN_WIDTH || imSize[0] > MAX_WIDTH) {
        LogError << "Invalid image width: " << imSize[0] << ". Width must be between " << MIN_WIDTH << " and "
                 << MAX_WIDTH << "." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    if (imSize[1] < MIN_HEIGHT || imSize[1] > MAX_HEIGHT) {
        LogError << "Invalid image height: " << imSize[1] << ". Height must be between " << MIN_HEIGHT << " and "
                 << MAX_HEIGHT << "." << GetErrorInfo(ERR_INVALID_PARAM);
        return ERR_INVALID_PARAM;
    }
    return SUCCESS;
}

ErrorCode ReadJpegData(const char* path, std::vector<uint8_t>& rawData, int& width, int& height,
                       std::shared_ptr<unsigned char[]>& decodedData)
{
    if (!IsFileValid(path)) {
        LogError << "Image path is invalid.";
        return ERR_INVALID_PARAM;
    }

    if (!CheckFileExtension(path, "jpg") && !CheckFileExtension(path, "jpeg")) {
        LogError << "Invalid image suffix, only support 'jpg', 'jpeg', 'JPG' or 'JPEG'.";
        return ERR_INVALID_PARAM;
    }

    // Buffer to hold the raw file data
    ErrorCode ret = ReadFile(path, rawData, IMAGE_MAX_FILE_SIZE);
    if (ret != SUCCESS) {
        return ret;
    }
    // Initialize JPEG decompressor
    tjhandle jpegDecompressor = tjInitDecompress();
    if (!jpegDecompressor) {
        LogError << "Image decompressor initialization failed. "
            << "Image decoding cannot proceed. Please check if the system has proper libjpeg-turbo support installed."
            << GetErrorInfo(ERR_LIBJPEG_INIT_FAILURE);
        return ERR_LIBJPEG_INIT_FAILURE;
    }

    int subSample;
    // Decompress header to get width, height, and subsampling
    int retInt = tjDecompressHeader2(jpegDecompressor, rawData.data(), rawData.size(), &width, &height, &subSample);
    if (retInt != 0) {
        LogError << "Invalid image data: failed to parse image header. "
                 << "Please ensure the input is a valid, non-corrupted image file."
                 << GetErrorInfo(ERR_LIBJPEG_READ_FILE_FAILURE);
        tjDestroy(jpegDecompressor);
        return ERR_LIBJPEG_READ_FILE_FAILURE;
    }
    // check image size before actual read data, prevent OOM.
    ret = CheckImSize({static_cast<size_t>(width), static_cast<size_t>(height)});
    if (ret != SUCCESS) {
        LogError << "Check image size failed." << GetErrorInfo(ret);
        tjDestroy(jpegDecompressor);
        return ret;
    }

    size_t totalBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * THREE_CHANNEL;
    decodedData = std::make_unique<unsigned char[]>(totalBytes);
    retInt = tjDecompress2(jpegDecompressor, rawData.data(), rawData.size(), decodedData.get(), width, 0, height,
                           TJPF_RGB, 0);
    if (retInt != 0) {
        LogError << "Invalid image data: failed to parse image data. "
                 << "Please ensure the input is a valid, non-corrupted image file."
                 << GetErrorInfo(ERR_LIBJPEG_READ_FILE_FAILURE);
        tjDestroy(jpegDecompressor);
        return ERR_LIBJPEG_READ_FILE_FAILURE;
    }
    tjDestroy(jpegDecompressor);
    return SUCCESS;
}
} // namespace Acc
