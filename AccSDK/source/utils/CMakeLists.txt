set(UTILS_SRC_LIST
        ${PROJECT_SOURCE_DIR}/source/utils/Log.cpp
        ${PROJECT_SOURCE_DIR}/source/utils/TensorUtils.cpp
        ${PROJECT_SOURCE_DIR}/source/utils/ThreadPool.cpp
        ${PROJECT_SOURCE_DIR}/source/utils/FileUtils.cpp
        ${PROJECT_SOURCE_DIR}/source/utils/ErrorCodeUtils.cpp
)

if(IMAGE OR VIDEO)
    list(APPEND UTILS_SRC_LIST
            ${PROJECT_SOURCE_DIR}/source/utils/ImageUtils.cpp
    )
endif()

if(VIDEO)
    list(APPEND UTILS_SRC_LIST
            ${PROJECT_SOURCE_DIR}/source/utils/VideoUtils.cpp
    )
endif()

target_sources(core PRIVATE ${UTILS_SRC_LIST})