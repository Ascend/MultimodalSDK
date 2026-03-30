cmake_minimum_required(VERSION 3.14.1)
project(ACCSDK)

# disable RPATH
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

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
        $ENV{ASCEND_TOOLKIT_HOME}/acllib/lib64
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

link_directories(${SECUREC_LIB_DIR})
set(SECUREC_LIB securec)

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