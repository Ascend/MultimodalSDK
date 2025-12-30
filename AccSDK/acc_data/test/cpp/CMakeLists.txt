# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
# Description: CMakeLists.txt
# Create: 2025-04-01
cmake_minimum_required(VERSION 3.14.1)

SET(TARGET_NAME  accdata_test)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# 包含所有组件的cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/conf/comp.cmake")

if (DEFINED ENV{BUILD_MODE})
    set(BUILD_MODE "$ENV{BUILD_MODE}")
else()
    set(BUILD_MODE "ut")
endif ()

set(ACCDATA_INTERFACE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../src/cpp/interface)
set(ACCDATA_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../../src/cpp)
set(SECUREC_DIR /usr/local/Ascend/driver)

SET(SECUREC_LIBS_DIR ${SECUREC_DIR}/lib64/driver/)
MACRO(TARGET_SECUREC target)
    TARGET_LINK_DIRECTORIES(${target} PUBLIC ${SECUREC_LIBS_DIR})
    TARGET_LINK_LIBRARIES(${target} PUBLIC libsecurec.so)
ENDMACRO()

if (NOT EXISTS ${GTEST_INSTALL_DIR})
    set(GTEST_INSTALL_DIR "${CMAKE_CURRENT_LIST_DIR}/../../output/3rdparty/gtest")
endif ()

include_directories(${ACCDATA_INTERFACE_DIR}
                    ${ACCDATA_SRC_DIR}
                    ${GTEST_INSTALL_DIR}/include
                    ${SECUREC_DIR}/include/dvpp/)

link_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../output/AccData/lib"
    ${GTEST_INSTALL_DIR}/lib
    ${GTEST_INSTALL_DIR}/lib64
)
add_link_options(-fsanitize=address)

if (${BUILD_MODE} MATCHES "ut")
    # 添加测试源代码
    file(GLOB_RECURSE ACCDATA_TEST_SOURCE_FILES
            ${CMAKE_CURRENT_LIST_DIR}/operator/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/pipeline/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/tensor/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/common/*.cpp
            )
elseif(${BUILD_MODE} MATCHES "fuzz")
    add_compile_options(-ftest-coverage -fprofile-arcs -fdump-rtl-expand)
    add_compile_options(-fsanitize=address -fsanitize-coverage=trace-pc -g -O0 -fPIC -fno-omit-frame-pointer)
    add_link_options(-fsanitize-recover=address)
    add_link_options(-fsanitize-address-use-after-scope)
    add_link_options(-fno-omit-frame-pointer)
    add_link_options(-fno-stack-protector)
    link_libraries(gcov)
    link_libraries(asan)
    # 添加测试源代码
    file(GLOB_RECURSE ACCDATA_TEST_SOURCE_FILES
            ${CMAKE_CURRENT_LIST_DIR}/fuzz/*.cpp
            )
endif ()

# 生成目标
add_executable(${TARGET_NAME} main.cpp ${ACCDATA_TEST_SOURCE_FILES})

# 打印构建选项
add_compile_options(-w)
get_target_property(COMPILE_FLAGS ${TARGET_NAME} COMPILE_OPTIONS)
get_target_property(LINK_FLAGS ${TARGET_NAME} LINK_OPTIONS)
target_link_libraries(${TARGET_NAME} PUBLIC gtest _accdata pthread)
TARGET_SECUREC(${TARGET_NAME})

message(STATUS "Compiler id: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compile flags: ${COMPILE_FLAGS}")
message(STATUS "Link flags: ${LINK_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")