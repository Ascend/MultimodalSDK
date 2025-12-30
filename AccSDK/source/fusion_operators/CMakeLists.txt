file(GLOB_RECURSE FUSION_OPERATOR_SOURCES
        ${PROJECT_SOURCE_DIR}/source/fusion_operators/*.cpp
)
target_sources(core PRIVATE ${FUSION_OPERATOR_SOURCES})