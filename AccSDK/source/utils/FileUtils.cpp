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
 * Description: File utils file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/utils/FileUtils.h"

#include <cstring>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"
#include "acc/utils/ErrorCodeUtils.h"
namespace fs = std::filesystem;

namespace {
constexpr int FILE_PATH_MAX = 4096;
constexpr mode_t FILE_MODE = 0640;
} // namespace

namespace Acc {
bool CheckFileExtension(const char* path, const char* suffix)
{
    // return if file is NULL or suffix is NULL
    if (!path || !suffix) {
        LogError << "Invalid file path, input file path or target suffix is nullptr.";
        return false;
    }
    // find char '.' reversely
    const char* pos = path + strlen(path);
    while (pos > path && *pos != '.') {
        pos--;
    }
    // not found
    if (pos == path) {
        LogError << "Invalid file path, input file has no suffix.";
        return false;
    }

    // start of extension skip char '.'
    const char* a = pos + 1;
    const char* b = suffix;
    // compare char by char
    bool compareResult = true;
    while (*a != '\0' && *b != '\0' && compareResult) {
        compareResult = (tolower(static_cast<unsigned char>(*a)) == tolower(static_cast<unsigned char>(*b)));
        if (compareResult) {
            a++;
            b++;
        }
    }
    // match when a and b both point to '\0'
    return (*a == '\0') && (*b == '\0');
}

ErrorCode ReadFile(const char* path, std::vector<uint8_t>& data, size_t maxFileSize)
{
    // open file
    fs::path normPath = fs::absolute(path).lexically_normal();
    std::ifstream file(normPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LogError << "Open file failed, file path is invalid" << GetErrorInfo(ERR_OPEN_FILE_FAILURE);
        return ERR_OPEN_FILE_FAILURE;
    }
    // get file size
    std::streampos pos = file.tellg();
    int64_t fileSize = (pos < 0) ? 0 : static_cast<int64_t>(pos);
    file.seekg(0, std::ios::beg);
    // check file is empty or not
    if (fileSize == 0) {
        LogError << "File is empty or failed to get size." << GetErrorInfo(ERR_INVALID_FILE_SIZE);
        return ERR_INVALID_FILE_SIZE;
    }
    // check large file
    if (fileSize > static_cast<int64_t>(maxFileSize)) {
        LogError << "File size exceeds maximum limit: " << fileSize << " bytes (max allowed: " << maxFileSize
                 << " bytes)." << GetErrorInfo(ERR_INVALID_FILE_SIZE);
        return ERR_INVALID_FILE_SIZE;
    }
    // read file data
    data.resize(fileSize);
    if (!file.read(reinterpret_cast<char*>(data.data()), fileSize)) {
        LogError << "Read file data failed." << GetErrorInfo(ERR_OPEN_FILE_FAILURE);
        return ERR_OPEN_FILE_FAILURE;
    }
    return SUCCESS;
}

bool CheckFilePath(const std::string& path)
{
    if (path.empty()) {
        LogError << "Check file path failed. The path is empty.";
        return false;
    }

    if (path.size() > FILE_PATH_MAX) {
        LogError << "Check file path failed. The file path size: " << path.size()
                 << " exceeds the maximum value: " << FILE_PATH_MAX << ".";
        return false;
    }

    fs::path pathObj = fs::absolute(path);
    // check file exits
    if (!fs::exists(pathObj)) {
        LogError << "Check file path failed. The file does not exist.";
        return false;
    }
    // check file is symlink or not
    if (fs::is_symlink(pathObj)) {
        LogError << "Check file path failed. The file is a symlink.";
        return false;
    }
    // check file is a regular file
    if (!fs::is_regular_file(pathObj)) {
        LogError << "Check file path failed. The file is not a regular file.";
        return false;
    }
    return true;
}

bool CheckFileOwner(const std::string& path)
{
    struct stat fileStat;
    // get file state information
    if (stat(path.c_str(), &fileStat) != 0) {
        LogError << "Check file owner failed, because get file stat failed.";
        return false;
    }

    // get user current id
    uid_t currentUid = getuid();
    if (fileStat.st_uid != currentUid) {
        LogError << "File owner mismatch. Process UID: " << currentUid << ", file UID: " << fileStat.st_uid << ".";
        return false;
    }
    return true;
}

bool CheckFilePermission(const std::string& path, const mode_t mode)
{
    struct stat buf;
    int ret = stat(path.c_str(), &buf);
    if (ret != 0) {
        LogError << "Check File Permission failed, because get file stat failed.";
        return false;
    }

    mode_t mask = 0700;
    const int perPermWidth = 3;
    std::vector<std::string> permMsg = {"Other group permission", "Owner group permission", "Owner permission"};
    for (int i = perPermWidth; i > 0; i--) {
        int curPerm = (buf.st_mode & mask) >> ((i - 1) * perPermWidth);
        int maxPerm = (mode & mask) >> ((i - 1) * perPermWidth);
        mask = mask >> perPermWidth;
        if (curPerm > maxPerm) {
            LogError << "Check " << permMsg[i - 1] << " failed: Current permission is " << curPerm
                     << ", but required no greater than " << maxPerm << ".";
            return false;
        }
    }
    return true;
}

bool IsFileValid(const char* path)
{
    if (!path) {
        LogError << "The file is invalid, check file path failed.";
        return false;
    }
    std::string pathStr(path);
    if (!CheckFilePath(pathStr)) {
        LogError << "The file is invalid, check file path failed.";
        return false;
    }
    if (!CheckFileOwner(pathStr)) {
        LogError << "The file is invalid, check file owner failed.";
        return false;
    }
    if (!CheckFilePermission(pathStr, FILE_MODE)) {
        LogError << "The file is invalid, check file permission failed.";
        return false;
    }
    return true;
}
} // namespace Acc