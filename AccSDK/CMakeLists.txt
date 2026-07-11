cmake_minimum_required(VERSION 3.14.1)
project(ACCSDK)

# disable RPATH
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# self include
include_directories(include)
include_directories(source/inc)
include_directories(source/py/include)

include_directories(
        $ENV{ASCEND_TOOLKIT_HOME}/acllib/include
        ${PROJECT_SOURCE_DIR}/opensource/libjpeg-turbo/include
        ${PROJECT_SOURCE_DIR}/opensource/FFmpeg/include
        ${PROJECT_SOURCE_DIR}/opensource/soxr/include
        ${PROJECT_SOURCE_DIR}/acc_data/src/cpp/interface/
)


link_directories(
        ${PROJECT_SOURCE_DIR}/opensource/libjpeg-turbo/lib
        ${PROJECT_SOURCE_DIR}/opensource/FFmpeg/lib
        ${PROJECT_SOURCE_DIR}/opensource/soxr/lib
)

set(SECUREC_PATH /usr/local/Ascend/driver)
file(GLOB_RECURSE SECUREC_LIB_FILES
        ${SECUREC_PATH}/libsecurec.so
)

if(NOT SECUREC_LIB_FILES)
    message(FATAL_ERROR "libsecurec.so not found under ${SECUREC_PATH}")
endif()

list(GET SECUREC_LIB_FILES 0 SECUREC_LIB_PATH)
get_filename_component(SECUREC_LIB_DIR ${SECUREC_LIB_PATH} DIRECTORY)

set(SECUREC_HEADER_SEARCH_DIRS
        ${SECUREC_PATH}
        /usr/local/Ascend/ascend-toolkit/latest
)
if(DEFINED ENV{ASCEND_HOME_PATH})
    list(APPEND SECUREC_HEADER_SEARCH_DIRS "$ENV{ASCEND_HOME_PATH}")
endif()
if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    list(APPEND SECUREC_HEADER_SEARCH_DIRS "$ENV{ASCEND_TOOLKIT_HOME}")
endif()

find_path(SECUREC_INCLUDE_DIR
        NAMES securec.h
        PATHS ${SECUREC_HEADER_SEARCH_DIRS}
        PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include runtime/include compiler/include
)
if(NOT SECUREC_INCLUDE_DIR)
    message(FATAL_ERROR "securec.h not found in Ascend driver or toolkit paths")
endif()

include_directories(${SECUREC_INCLUDE_DIR})
link_directories(${SECUREC_LIB_DIR})
set(SECUREC_LIB securec)

set(ASCENDCL_SEARCH_DIRS
        /usr/local/Ascend/ascend-toolkit/latest
)
if(DEFINED ENV{ASCEND_HOME_PATH})
    list(APPEND ASCENDCL_SEARCH_DIRS "$ENV{ASCEND_HOME_PATH}")
endif()
if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    list(APPEND ASCENDCL_SEARCH_DIRS "$ENV{ASCEND_TOOLKIT_HOME}")
endif()

# CANN 8 uses acllib/lib64 while CANN 9 places AscendCL under the architecture
# specific toolkit directory.
find_library(ASCENDCL_LIBRARY
        NAMES ascendcl
        PATHS ${ASCENDCL_SEARCH_DIRS}
        PATH_SUFFIXES acllib/lib64 aarch64-linux/lib64 x86_64-linux/lib64 lib64 runtime/lib64
)
if(NOT ASCENDCL_LIBRARY)
    message(FATAL_ERROR "libascendcl.so not found in Ascend toolkit paths")
endif()
get_filename_component(ASCENDCL_LIBRARY_DIR ${ASCENDCL_LIBRARY} DIRECTORY)
link_directories(${ASCENDCL_LIBRARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--copy-dt-needed-entries")
add_compile_options(-std=c++17 -fPIE -fstack-protector-all -fPIC -Wall -Wextra -Wfloat-equal -pipe)

if(BUILD_TESTS)
    include_directories("$ENV{GTEST_HOME}/include")
    link_directories("$ENV{GTEST_HOME}/../build/lib")
    add_compile_options(-g)
    add_compile_options(--coverage -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack -pie --coverage)
    enable_testing()
    add_subdirectory(test)
else()
    add_compile_options(-O2)
    add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie)
endif()

add_subdirectory(source)
