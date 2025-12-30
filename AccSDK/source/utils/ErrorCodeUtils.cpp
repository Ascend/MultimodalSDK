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
 * Description: Error code utils.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/utils/ErrorCodeUtils.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
namespace Acc {
namespace {
const size_t CODE_WIDTH = 8;
const std::vector<std::string> INVALID_CHAR = {"\n",     "\f",     "\r",     "\b",     "\t",     "\v",    "\u000D",
                                               "\u000A", "\u000C", "\u000B", "\u0009", "\u0008", "\u007F"};

const std::string DATA_CHECK_ERROR_INFO_STRING[] = {
    [0] = "Undefined error code",
    [static_cast<uint32_t>(SubErrorCode::INVALID_PARAM) - static_cast<uint32_t>(SubErrorCode::DATA_CHECK_ERROR_BEGIN)] =
        "Invalid parameter",
    [static_cast<uint32_t>(SubErrorCode::UNSUPPORTED_TYPE) -
        static_cast<uint32_t>(SubErrorCode::DATA_CHECK_ERROR_BEGIN)] = "Unsupported type",
    [static_cast<uint32_t>(SubErrorCode::INVALID_POINTER) -
        static_cast<uint32_t>(SubErrorCode::DATA_CHECK_ERROR_BEGIN)] = "Invalid pointer",
    [static_cast<uint32_t>(SubErrorCode::OUT_OF_RANGE) - static_cast<uint32_t>(SubErrorCode::DATA_CHECK_ERROR_BEGIN)] =
        "Out of range",
};

const std::string RESOURCE_ERROR_INFO_STRING[] = {
    [0] = "Undefined error code",
    [static_cast<uint32_t>(SubErrorCode::BAD_ALLOC) - static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] =
        "Memory allocation failed",
    [static_cast<uint32_t>(SubErrorCode::BAD_COPY) - static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] =
        "Memory copy failed",
    [static_cast<uint32_t>(SubErrorCode::BAD_FREE) - static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] =
        "Memory free failed",
    [static_cast<uint32_t>(SubErrorCode::OUT_OF_MEM) - static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] =
        "Out of memory",
    [static_cast<uint32_t>(SubErrorCode::OPEN_FILE_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] = "Open file failed",
    [static_cast<uint32_t>(SubErrorCode::INVALID_FILE_PERMISSION) -
        static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] = "Invalid file permission",
    [static_cast<uint32_t>(SubErrorCode::INVALID_FILE_SIZE) -
        static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN)] = "Invalid file size",
};

const std::string OPENSOURCE_ERROR_INFO_STRING[] = {
    [0] = "Undefined error code",
    [static_cast<uint32_t>(SubErrorCode::FFMPEG_INIT_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::OPENSOURCE_ERROR_BEGIN)] = "FFMPEG init failed",
    [static_cast<uint32_t>(SubErrorCode::LIBJPEG_INIT_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::OPENSOURCE_ERROR_BEGIN)] = "LibJpeg init failed",
    [static_cast<uint32_t>(SubErrorCode::LIBJPEG_READ_FILE_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::OPENSOURCE_ERROR_BEGIN)] = "LibJpeg read file failed",
    [static_cast<uint32_t>(SubErrorCode::FFMPEG_COMMON_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::OPENSOURCE_ERROR_BEGIN)] = "FFmpeg inner function execute failed",
};

const std::string THIRD_PARTY_ERROR_INFO_STRING[] = {
    [0] = "Undefined error code",
    [static_cast<uint32_t>(SubErrorCode::ACC_DATA_INIT_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::THIRD_PARTY_ERROR_BEGIN)] = "Inner operators failure",
    [static_cast<uint32_t>(SubErrorCode::ACC_DATA_EXECUTE_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::THIRD_PARTY_ERROR_BEGIN)] = "Inner functions execute failure",
    [static_cast<uint32_t>(SubErrorCode::ACC_DATA_PROPERTY_CONVERT_FAILURE) -
        static_cast<uint32_t>(SubErrorCode::THIRD_PARTY_ERROR_BEGIN)] = "Inner type convert failure",
};

const std::string RUNTIME_ERROR_INFO_STRING[] = {
    [0] = "Undefined error code",
    [static_cast<uint32_t>(SubErrorCode::WAIT_TIME_OUT) - static_cast<uint32_t>(SubErrorCode::RUNTIME_ERROR_BEGIN)] =
        "Waiting timed out",
    [static_cast<uint32_t>(SubErrorCode::INVALID_THREAD_POOL_STATUS) -
        static_cast<uint32_t>(SubErrorCode::RUNTIME_ERROR_BEGIN)] = "Invalid thread pool status",
};

template<typename T>
uint32_t GetArrayLen(const T& arr)
{
    return (sizeof(arr) / sizeof(arr[0]));
}

const std::pair<const std::string*, uint32_t> ERROR_TYPE_TO_ERROR_INFO[] = {
    [0] = {nullptr, 0},
    [static_cast<uint32_t>(ErrorType::DATA_CHECK_ERROR)] = {DATA_CHECK_ERROR_INFO_STRING,
                                                            GetArrayLen(DATA_CHECK_ERROR_INFO_STRING)},
    [static_cast<uint32_t>(ErrorType::RESOURCE_ERROR)] = {RESOURCE_ERROR_INFO_STRING,
                                                          GetArrayLen(RESOURCE_ERROR_INFO_STRING)},
    [static_cast<uint32_t>(ErrorType::OPENSOURCE_ERROR)] = {OPENSOURCE_ERROR_INFO_STRING,
                                                            GetArrayLen(OPENSOURCE_ERROR_INFO_STRING)},
    [static_cast<uint32_t>(ErrorType::THIRD_PARTY_ERROR)] = {THIRD_PARTY_ERROR_INFO_STRING,
                                                             GetArrayLen(THIRD_PARTY_ERROR_INFO_STRING)},
    [static_cast<uint32_t>(ErrorType::RUNTIME_ERROR)] = {RUNTIME_ERROR_INFO_STRING,
                                                         GetArrayLen(RUNTIME_ERROR_INFO_STRING)},
};

void ReplaceInvalidChar(std::string& text)
{
    for (auto& filter : INVALID_CHAR) {
        if (text.find(filter) == std::string::npos) {
            continue;
        }
        std::string::size_type pos = 0;
        while ((pos = text.find(filter)) != std::string::npos) {
            text.replace(pos, filter.length(), " ");
        }
    }

    for (size_t i = text.size() - 1; i > 0; i--) {
        if (text[i] == text[i - 1] && text[i] == ' ') {
            text.erase(text.begin() + i);
        }
    }
}

std::string GetErrorCodeInfo(const ErrorCode errorCode)
{
    if (errorCode == SUCCESS) {
        return "Success";
    }
    std::string retInfo = "Undefined error code";
    uint32_t errorType = static_cast<uint32_t>((errorCode >> 20) & 0xFF);
    const std::string* typeArr =
        (errorType < GetArrayLen(ERROR_TYPE_TO_ERROR_INFO)) ? ERROR_TYPE_TO_ERROR_INFO[errorType].first : nullptr;
    if (!typeArr) {
        return retInfo;
    }
    uint32_t code = static_cast<uint32_t>(errorCode & 0xFFFFF);
    uint32_t codeOffset = code - (errorType - 1) * RANGE_SIZE;
    retInfo = (codeOffset < ERROR_TYPE_TO_ERROR_INFO[errorType].second) ? typeArr[codeOffset] : retInfo;
    return retInfo;
}
} // namespace

std::string GetErrorInfo(ErrorCode err, std::string callingFuncName)
{
    std::stringstream errorInfo;
    errorInfo << " (";
    // add calling function information
    if (!callingFuncName.empty()) {
        ReplaceInvalidChar(callingFuncName);
        errorInfo << "Calling Function = " << callingFuncName << ", ";
    }
    // add error code information
    std::stringstream codeSs;
    codeSs << std::uppercase << std::setfill('0') << std::setw(CODE_WIDTH) << std::hex << err;
    std::string codeStr = codeSs.str();
    std::transform(codeStr.begin(), codeStr.end(), codeStr.begin(), ::toupper);
    errorInfo << "Code = "
              << "0x" << codeStr << ", ";
    // add error message information
    errorInfo << "Message = "
              << "\"" << GetErrorCodeInfo(err) << "\") ";
    return errorInfo.str();
}
} // namespace Acc