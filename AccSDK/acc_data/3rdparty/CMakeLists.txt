add_subdirectory(pybind)

if (${CMAKE_BUILD_TYPE} MATCHES "Ut" OR ${CMAKE_BUILD_TYPE} MATCHES "Fuzz")
    add_subdirectory(gtest)
endif (${CMAKE_BUILD_TYPE} MATCHES "Ut")