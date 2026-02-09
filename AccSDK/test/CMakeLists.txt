project(accGtestUnitCases)

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(tensor)
add_subdirectory(py)
add_subdirectory(fusion_operators)


if(IMAGE)
    add_subdirectory(image)
endif()
if(VIDEO)
    add_subdirectory(video)
endif()
if(AUDIO)
    add_subdirectory(audio)
endif()