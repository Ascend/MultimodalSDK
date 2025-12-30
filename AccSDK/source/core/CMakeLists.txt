file(GLOB_RECURSE CORE_SOURCES
        ${PROJECT_SOURCE_DIR}/source/core/framework/*.cpp
)
target_sources(core PRIVATE ${CORE_SOURCES})