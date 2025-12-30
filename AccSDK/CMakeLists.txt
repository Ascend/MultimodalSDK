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
        ${PROJECT_SOURCE_DIR}/acc_data/src/cpp/interface/
)


link_directories(
        $ENV{ASCEND_TOOLKIT_HOME}/acllib/lib64
        ${PROJECT_SOURCE_DIR}/opensource/libjpeg-turbo/lib
        ${PROJECT_SOURCE_DIR}/opensource/FFmpeg/lib
)

set(SECUREC_PATH1 "$ENV{ASCEND_TOOLKIT_HOME}/../../driver/lib64/driver")
set(SECUREC_PATH2 "$ENV{ASCEND_TOOLKIT_HOME}/../../driver/lib64/common")

if(EXISTS "${SECUREC_PATH1}/libsecurec.so")
    link_directories("${SECUREC_PATH1}")
    set(SECUREC_LIB securec)
elseif(EXISTS "${SECUREC_PATH2}/libsecurec.so")
    link_directories("${SECUREC_PATH2}")
    set(SECUREC_LIB securec)
else()
    message(FATAL_ERROR "libsecurec.so not found in expected paths.")
endif()

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