file(GLOB_RECURSE IMAGE_SOURCES
        ${PROJECT_SOURCE_DIR}/source/image/ops/*.cpp
        ${PROJECT_SOURCE_DIR}/source/image/*.cpp
)

target_sources(core PRIVATE ${IMAGE_SOURCES})
