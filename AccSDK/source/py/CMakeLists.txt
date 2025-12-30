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
add_definitions(-DPy_LIMITED_API=0x03090000)
find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Interpreter Development)

find_package(SWIG REQUIRED)
include(UseSWIG)
find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Interpreter Development)

include_directories(${PROJECT_SOURCE_DIR}/source/py/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(SWIG_INTERFACE_FILE ${PROJECT_SOURCE_DIR}/source/py/SwigInterface/acc.i)

set_source_files_properties(${SWIG_INTERFACE_FILE} PROPERTIES CPLUSPLUS ON)

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
)

target_include_directories(acc PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/source/py/include
    ${PROJECT_SOURCE_DIR}/include
)

swig_link_libraries(acc
    PRIVATE ascendcl core
)