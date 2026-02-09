file(GLOB_RECURSE AUDIO_SOURCES
            ${PROJECT_SOURCE_DIR}/source/audio/*.cpp
)
target_sources(core PRIVATE ${AUDIO_SOURCES})