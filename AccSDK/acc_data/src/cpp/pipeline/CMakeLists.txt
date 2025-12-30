add_subdirectory(workspace)
add_subdirectory(executor)
add_subdirectory(graph)

aux_source_directory(./ INPUT_PIPELINE_FILES)

target_sources(_accdata PRIVATE ${INPUT_PIPELINE_FILES})