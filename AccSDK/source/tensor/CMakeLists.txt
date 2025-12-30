file(GLOB_RECURSE TENSOR_SOURCES
        ${PROJECT_SOURCE_DIR}/source/tensor/checker/*.cpp
        ${PROJECT_SOURCE_DIR}/source/tensor/*.cpp
)

target_sources(core PRIVATE ${TENSOR_SOURCES})