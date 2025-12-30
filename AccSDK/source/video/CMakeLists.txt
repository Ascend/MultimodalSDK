file(GLOB_RECURSE VIDEO_SOURCES
                ${PROJECT_SOURCE_DIR}/source/video/*.cpp
)

target_sources(core PRIVATE ${VIDEO_SOURCES})