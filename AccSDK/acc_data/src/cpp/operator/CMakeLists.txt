add_subdirectory(input)
add_subdirectory(image)
add_subdirectory(math)
add_subdirectory(fusion)

aux_source_directory(./ OP_FILES)

target_sources(_accdata PRIVATE ${OP_FILES})