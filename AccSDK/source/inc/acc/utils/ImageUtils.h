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
 * Description: Internal image utils header file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <memory>
#include "acc/ErrorCode.h"

namespace Acc {
/**
 * @description: Read jpeg image meta data and optionally decode the full image.
 * @param path: Input jpeg image path.
 * @param raw_data: Output jpeg source data (filled if onlyParseHeader is true).
 * @param width: Output jpeg image width.
 * @param height: Output jpeg image height.
 * @param decoded_data: Output decoded RGB data as a shared_ptr<unsigned char[]>
 * (filled if onlyParseHeader is false).
 * @param onlyParseHeader: If true, only reads metadata and fills raw_data;
 * otherwise, decodes the image and fills decoded_data.
 * @return: int, Error code (SUCCESS or an error code).
 */
ErrorCode ReadJpegData(const char* path, std::vector<uint8_t>& rawData, int& width, int& height,
                       std::shared_ptr<unsigned char[]>& decodedData);

/**
 * @description: Check image size.
 * @param vector<size_t>: Image size.
 * @return: int, Error code (SUCCESS or an error code).
 */
ErrorCode CheckImSize(const std::vector<size_t>& imSize);
} // namespace Acc

#endif // IMAGE_UTILS_H