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
* Description: Internal file utils header file.
* Author: ACC SDK
* Create: 2025
* History: NA
*/
#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <sys/types.h>
#include <vector>
#include <memory>
#include "acc/ErrorCode.h"

namespace Acc {
inline constexpr size_t DEFAULT_MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1GB
/**
* @description: Check whether file extension match the target.
* @param path: Input file path.
* @param suffix: Input target file extension.
* @return: Bool
*/
bool CheckFileExtension(const char* path, const char* suffix);

/**
* @description: Read file source data.
* @param path: Input file path.
* @param data: Output file data.
* @return: int, Error code.
*/
ErrorCode ReadFile(const char* path, std::vector<uint8_t>& data, size_t maxFileSize = DEFAULT_MAX_FILE_SIZE);

/**
* @description: Check file path size、symlink、regular file
* @param path: File path
* @return: Bool
*/
bool CheckFilePath(const std::string& path);

/**
* @description: Check file owner
* @param path: File path
* @return: Bool
*/
bool CheckFileOwner(const std::string& path);

/**
* @description: Check file permission no greater than specified mode
* @param path: File path
* @param mode: specified mode
* @return: Bool
*/
bool CheckFilePermission(const std::string& path, const mode_t mode);

/**
* @description: Check file valid, including file path, owner and permission
* @param path: File path
* @return: Bool
*/
bool IsFileValid(const char* path);
} // namespace Acc

#endif // FILE_UTILS_H
