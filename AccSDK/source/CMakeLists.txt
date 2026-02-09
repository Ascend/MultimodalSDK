add_library(core SHARED)

add_subdirectory(core)
add_subdirectory(py)
add_subdirectory(tensor)
add_subdirectory(utils)
add_subdirectory(fusion_operators)

if(IMAGE OR VIDEO)
    add_subdirectory(image)
endif()
if(VIDEO)
    add_subdirectory(video)
endif()
if(AUDIO)
    add_subdirectory(audio)
endif()
target_link_libraries(core
        PRIVATE
        -Wl,--no-as-needed
        ascendcl
        pthread
        ${SECUREC_LIB}
        -Wl,--whole-archive
        "${PROJECT_SOURCE_DIR}/acc_data/output/AccData/lib/lib_accdata.a"
        -Wl,--no-whole-archive
)
if(IMAGE OR VIDEO)
    target_link_libraries(core
            PRIVATE
            turbojpeg)
endif()

if(VIDEO)
    target_link_libraries(core PRIVATE avformat avcodec avutil swscale)
endif()
if(AUDIO)
    target_link_libraries(core PRIVATE soxr)
endif()

install(TARGETS core DESTINATION ${CMAKE_BINARY_DIR}/lib)