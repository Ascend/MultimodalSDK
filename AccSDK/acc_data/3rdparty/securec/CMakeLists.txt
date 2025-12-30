set(SECUREC_SOURCE_DIR ${PROJECT_3RDPARTY_SRC_DIR}/securec/huawei_secure_c)
set(SECUREC_BUILD_DIR ${PROJECT_BUILD_PATH}/3rdparty/securec)
set(SECUREC_INSTALL_DIR ${PROJECT_3RDPARTY_BIN_DIR}/securec)

set(CHECK_FILE_SECUREC ${SECUREC_INSTALL_DIR}/lib/libsecurec.so)

if(NOT EXISTS  ${CHECK_FILE_SECUREC})
    message("build securec")
    # create build dir
    exec_program(mkdir ${PROJECT_BUILD_PATH} ARGS -p ${SECUREC_BUILD_DIR})
    exec_program(mkdir ${PROJECT_3RDPARTY_BIN_DIR} ARGS -p ${SECUREC_INSTALL_DIR})

    # execute make && make install
    exec_program(make ${SECUREC_SOURCE_DIR}/src ARGS -j)
    exec_program(cp ARGS -r ${SECUREC_SOURCE_DIR}/include ${SECUREC_INSTALL_DIR})
    exec_program(cp ARGS -r ${SECUREC_SOURCE_DIR}/lib ${SECUREC_INSTALL_DIR})
else ()
    message("-- securec: ${CHECK_FILE_SECUREC}")
    message("-- securec: has been built, ignored")
endif(NOT EXISTS ${CHECK_FILE_SECUREC})