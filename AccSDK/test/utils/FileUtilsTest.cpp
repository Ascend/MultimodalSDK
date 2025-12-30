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
 * Description:  file utils api.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <pwd.h>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gtest/gtest.h>
#include <acc/utils/FileUtils.h>
#include "acc/ErrorCode.h"

static int (*g_realStat)(const char*, struct stat*) = nullptr;

extern "C" int __xstat(int ver, const char* path, struct stat* buf)
{
    static int (*real_xstat)(int, const char*, struct stat*) = nullptr;
    if (!real_xstat) {
        real_xstat = (decltype(real_xstat))dlsym(RTLD_NEXT, "__xstat");
    }
    return real_xstat(ver, path, buf);
}

using namespace Acc;
namespace {
constexpr int FILE_PATH_MAX = 4096;
constexpr mode_t MOCK_MODE = 0755;
const std::string EXPECT_FILE_CONTENT = "Hello, World!";
class FileUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        // Create temporary files for testing
        std::ofstream outfile("valid_file.bin", std::ios::binary);
        outfile << EXPECT_FILE_CONTENT;
        outfile.close();
        std::ofstream outfile2("empty_file.bin", std::ios::binary);
        outfile2.close();

        std::ofstream("regular_file.txt").close();
        std::ofstream("symlink.txt").close();
        std::remove("symlink.txt");
        symlink("regular_file.txt", "symlink.txt");
    }

    void TearDown() override
    {
        // Remove temporary files
        std::remove("valid_file.bin");
        std::remove("empty_file.bin");
        std::remove("regular_file.txt");
        std::remove("symlink.txt");
    }
};
TEST_F(FileUtilsTest, CheckFileExtension_ShouldReturnTrue_WhenPathIsValid)
{
    const char* path = "test.jpg";
    bool ret = CheckFileExtension(path, "jpg");
    EXPECT_TRUE(ret);
}

TEST_F(FileUtilsTest, CheckFileExtension_ShouldReturnFalse_WhenPathIsNullptr)
{
    bool ret = CheckFileExtension(nullptr, "jpg");
    EXPECT_EQ(ret, false);
}

TEST_F(FileUtilsTest, CheckFileExtension_ShouldReturnFalse_WhenPathNoSuffix)
{
    const char* path = "test";
    bool ret = CheckFileExtension(path, "jpg");
    EXPECT_EQ(ret, false);
}

TEST_F(FileUtilsTest, ReadFile_ShouldReturnSuccess_WhenFileReadSucceeds)
{
    std::vector<uint8_t> data;
    ErrorCode ret = ReadFile("valid_file.bin", data);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(data.size(), EXPECT_FILE_CONTENT.size());
    if (data.size() >= EXPECT_FILE_CONTENT.size()) {
        const std::vector<uint8_t> expectedData(EXPECT_FILE_CONTENT.begin(), EXPECT_FILE_CONTENT.end());
        EXPECT_EQ(data, expectedData);
    }
}

TEST_F(FileUtilsTest, ReadFile_ShouldReturnErr_WhenPathIsInvalid)
{
    std::vector<uint8_t> data;
    ErrorCode ret = ReadFile("invalid_path.bin", data);
    EXPECT_EQ(ret, ERR_OPEN_FILE_FAILURE);
}

TEST_F(FileUtilsTest, ReadFile_ShouldReturnErr_WhenFileIsEmpty)
{
    std::vector<uint8_t> data;
    ErrorCode ret = ReadFile("empty_file.bin", data);
    EXPECT_EQ(ret, ERR_INVALID_FILE_SIZE);
}

TEST_F(FileUtilsTest, ReadFile_ShouldReturnErr_WhenFileIsTooLarge)
{
    const char* testFilePath = "test_2mb_file.bin";
    const size_t maxSize = 1 * 1024 * 1024;
    {
        std::ofstream ofs(testFilePath, std::ios::binary);
        std::vector<char> buffer(1024 * 1024, 0xAA); // 1MB buffer with 0xAA
        ofs.write(buffer.data(), buffer.size());
        ofs.write(buffer.data(), buffer.size());
        ofs.close();
    }

    std::vector<uint8_t> data;
    ErrorCode ret = ReadFile(testFilePath, data, maxSize);
    EXPECT_EQ(ret, ERR_INVALID_FILE_SIZE);
    std::remove(testFilePath);
}

TEST_F(FileUtilsTest, CheckFilePath_ShouldReturnFalse_WhenPathIsEmpty)
{
    std::string emptyPath = "";
    EXPECT_FALSE(CheckFilePath(emptyPath));
}

TEST_F(FileUtilsTest, CheckFilePath_ShouldReturnFalse_WhenPathExceedsMaxLength)
{
    std::string longPath(FILE_PATH_MAX + 1, 'a');
    EXPECT_FALSE(CheckFilePath(longPath));
}

TEST_F(FileUtilsTest, CheckFilePath_ShouldReturnFalse_WhenPathDoesNotExist)
{
    std::string nonExistentPath = "non_existent_file.txt";
    EXPECT_FALSE(CheckFilePath(nonExistentPath));
}

TEST_F(FileUtilsTest, CheckFilePath_ShouldReturnFalse_WhenPathIsSymlink)
{
    std::string symlinkPath = "symlink.txt";
    EXPECT_FALSE(CheckFilePath(symlinkPath));
}

TEST_F(FileUtilsTest, CheckFilePath_ShouldReturnFalse_WhenPathIsNotRegularFile)
{
    std::string directoryPath = "."; // Current directory is not a regular file
    EXPECT_FALSE(CheckFilePath(directoryPath));
}

TEST_F(FileUtilsTest, CheckFilePath_ShouldReturnTrue_WhenPathIsValid)
{
    std::string validPath = "regular_file.txt";
    EXPECT_TRUE(CheckFilePath(validPath));
}

TEST_F(FileUtilsTest, CheckFileOwner_ShouldReturnFalse_WhenStatFails)
{
    std::string invalidPath = "/path/to/nonexistent/file";
    bool result = CheckFileOwner(invalidPath);
    EXPECT_FALSE(result);
}

TEST_F(FileUtilsTest, CheckFileOwner_ShouldReturnFalse_WhenFileOwnerMismatch)
{
    ASSERT_NO_THROW({
        // create other owner file
        std::string filePath = "test.txt";
        std::string targetUser = "HwHiAiUser";
        struct passwd* pwd = getpwnam(targetUser.c_str());
        if (pwd == nullptr) {
            throw std::runtime_error("User not found: " + targetUser);
        }
        int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            throw std::runtime_error("Failed to create file: " + filePath + ", error: " + std::string(strerror(errno)));
        }
        uid_t targetUid = pwd->pw_uid;
        if (chown(filePath.c_str(), targetUid, -1) != 0) {
            close(fd);
            unlink(filePath.c_str());
            throw std::runtime_error("Failed to change file owner: " + filePath +
                                     ", error: " + std::string(strerror(errno)));
        }

        // check file owner
        bool ret = CheckFileOwner(filePath);
        EXPECT_FALSE(ret);
        close(fd);
        std::remove("test.txt");
    });
}

TEST_F(FileUtilsTest, CheckFileOwner_ShouldReturnTrue_WhenFileOwnerMatches)
{
    std::string filePath = "valid_file.bin";
    bool ret = CheckFileOwner(filePath);
    EXPECT_TRUE(ret);
}

TEST_F(FileUtilsTest, CheckFilePermission_ShouldReturnFalse_WhenStatCallFails)
{
    std::string invalidPath = "invalid_path.bin";
    mode_t mode = 0700;
    EXPECT_FALSE(CheckFilePermission(invalidPath, mode));
}

TEST_F(FileUtilsTest, CheckFilePermission_ShouldReturnFalse_WhenFilePermissionExceedsMode)
{
    std::string filePath = "valid_file.bin";
    mode_t mode = 0700;

    // save origin function pointer
    auto oldStat = g_realStat;

    // set mock function
    g_realStat = [](const char*, struct stat* buf) -> int {
        buf->st_mode = MOCK_MODE;
        return 0;
    };

    EXPECT_FALSE(CheckFilePermission(filePath, mode));

    // restore origin function pointer
    g_realStat = oldStat;
}

TEST_F(FileUtilsTest, CheckFilePermission_ShouldReturnTrue_WhenFilePermissionDoesNotExceedMode)
{
    std::string filePath = "valid_file.bin";
    mode_t mode = 0755;

    // save origin function pointer
    auto oldStat = g_realStat;

    // set mock function
    g_realStat = [](const char*, struct stat* buf) -> int {
        buf->st_mode = MOCK_MODE;
        return 0;
    };

    EXPECT_TRUE(CheckFilePermission(filePath, mode));

    // restore origin function pointer
    g_realStat = oldStat;
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
