#build secure
set(SOURCE_DIR ${PROJECT_3RDPARTY_SRC_DIR}/gtest/googletest)
set(BUILD_DIR ${PROJECT_BUILD_PATH}/3rdparty/gtest/)
set(INSTALL_DIR ${PROJECT_3RDPARTY_BIN_DIR}/gtest)

set(CHECK_FILE_GTEST ${INSTALL_DIR}/lib64/libgtest.a)
message("-- gtest: check if need to build")

# check if the file exists or not, if not build it
if(NOT EXISTS  ${CHECK_FILE_GTEST})
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_STANDARD 17)

    # create build dir
    exec_program(mkdir ${BUILD_DIR} ARGS -p ${BUILD_DIR})
    exec_program(mkdir ${BUILD_DIR} ARGS -p ${INSTALL_DIR})

    exec_program(cmake ${BUILD_DIR} ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" ${SOURCE_DIR})

    # execute make && make install
    exec_program(make ${BUILD_DIR} ARGS clean)
    exec_program(make ${BUILD_DIR} ARGS -j8)
    exec_program(make ${BUILD_DIR} ARGS install)
else()
    message("-- gtest: ${CHECK_FILE_GTEST}")
    message("-- gtest: has been built, ignored")
endif(NOT EXISTS ${CHECK_FILE_GTEST})
