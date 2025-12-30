#build secure
set(SOURCE_DIR ${PROJECT_3RDPARTY_SRC_DIR}/pybind/pybind11)
set(BUILD_DIR ${PROJECT_BUILD_PATH}/3rdparty/pybind11/)
set(INSTALL_DIR ${PROJECT_3RDPARTY_BIN_DIR}/pybind11)

set(CHECK_FILE_PYBIND ${INSTALL_DIR}/include/pybind11/pytypes.h)
message("-- pybind: check if need to build")

# check if the file exists or not, if not build it
if(NOT EXISTS  ${CHECK_FILE_PYBIND})
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_STANDARD 11)

    # create build dir
    exec_program(mkdir ${BUILD_DIR} ARGS -p ${BUILD_DIR})
    exec_program(mkdir ${BUILD_DIR} ARGS -p ${INSTALL_DIR})

    exec_program(cmake ${BUILD_DIR} ARGS  -DPYBIND11_NOPYTHON=ON -DPYBIND11_TEST=OFF -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" ${SOURCE_DIR})

    # execute make && make install
    exec_program(make ${BUILD_DIR} ARGS clean)
    exec_program(make ${BUILD_DIR} ARGS -j8)
    exec_program(make ${BUILD_DIR} ARGS install)
else()
    message("-- pybind: ${CHECK_FILE_PYBIND}")
    message("-- pybind: has been built, ignored")
endif(NOT EXISTS ${CHECK_FILE_PYBIND})

set(ENV{CMAKE_PREFIX_PATH} ${INSTALL_DIR})
#set(ENV{CMAKE_PREFIX_PATH} "$ENV{CMAKE_PREFIX_PATH};${INSTALL_DIR}")