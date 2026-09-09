set(PYTHON_SRC_LIST
    ${PROJECT_SOURCE_DIR}/source/py/module/PyTensor.cpp
    ${PROJECT_SOURCE_DIR}/source/py/module/PyTensorOps.cpp
)

if(IMAGE OR VIDEO)
    add_definitions(-DENABLE_IMAGE)
    list(APPEND PYTHON_SRC_LIST
        ${PROJECT_SOURCE_DIR}/source/py/module/PyImage.cpp
    )
endif()

if(VIDEO)
    add_definitions(-DENABLE_VIDEO)
    list(APPEND PYTHON_SRC_LIST
        ${PROJECT_SOURCE_DIR}/source/py/module/PyVideo.cpp
    )
endif()

if(AUDIO)
    add_definitions(-DENABLE_AUDIO)
    list(APPEND PYTHON_SRC_LIST
            ${PROJECT_SOURCE_DIR}/source/py/module/PyAudio.cpp
    )
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)

find_package(SWIG REQUIRED)
include(UseSWIG)

include_directories(${PROJECT_SOURCE_DIR}/source/py/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(SWIG_INTERFACE_FILE ${PROJECT_SOURCE_DIR}/source/py/SwigInterface/acc.i)

set_source_files_properties(${SWIG_INTERFACE_FILE} PROPERTIES CPLUSPLUS ON)

# 开启SWIG线程支持：wrapper进入C++前释放GIL，Director回调Python前自动获取GIL，
# 保证线程池worker线程（如音频批量解码）回调Python日志函数时不会产生数据竞争或死锁。
# 注意：必须用CMAKE_SWIG_FLAGS（或源文件属性COMPILE_OPTIONS / target属性SWIG_COMPILE_OPTIONS），
# 源文件属性不支持SWIG_COMPILE_OPTIONS，会被CMake静默忽略。
# 先检查防止重复添加，用append避免覆盖外部可能已设置的CMAKE_SWIG_FLAGS。
if(NOT "-threads" IN_LIST CMAKE_SWIG_FLAGS)
    list(APPEND CMAKE_SWIG_FLAGS "-threads")
endif()

swig_add_library(acc
    TYPE MODULE
    LANGUAGE python
    SOURCES
        ${SWIG_INTERFACE_FILE}
        ${PROJECT_SOURCE_DIR}/source/py/module/PyTensor.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyTensorOps.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyImage.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyUtil.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyLog.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyPreprocess.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyVideo.cpp
        ${PROJECT_SOURCE_DIR}/source/py/module/PyAudio.cpp
)

target_include_directories(acc PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/source/py/include
    ${PROJECT_SOURCE_DIR}/include
)

swig_link_libraries(acc
    PRIVATE ascendcl core
)
