find_package(pybind11 REQUIRED)

if (NOT DEFINED pybind11_FOUND)
    message(FATAL_ERROR "use python11 build, but pybind11 module is missing!")
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES "Ut")
    add_compile_options(-ftest-coverage -fprofile-arcs -fdump-rtl-expand)
    add_compile_options(-fsanitize=address -fsanitize-coverage=trace-pc -g -O0 -fPIC -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    link_libraries(gcov)
endif (${CMAKE_BUILD_TYPE} MATCHES "Ut")

if (${CMAKE_BUILD_TYPE} MATCHES "Fuzz")
    add_compile_options(-ftest-coverage -fprofile-arcs -fdump-rtl-expand)
    add_compile_options(-fsanitize=address -fsanitize-coverage=trace-pc -g -O0 -fPIC -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    add_link_options(-fsanitize-recover=address)
    add_link_options(-fsanitize-address-use-after-scope)
    add_link_options(-fno-omit-frame-pointer)
    add_link_options(-fno-stack-protector)
    link_libraries(gcov)
    link_libraries(asan)
endif (${CMAKE_BUILD_TYPE} MATCHES "Fuzz")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/interface)

add_library(_accdata SHARED)

target_link_libraries(_accdata PUBLIC ${PROJECT_SECUREC_DIR}/lib64/driver/libsecurec.so)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
# All source files in these subdirectories will be add to accdata
add_subdirectory(common)
add_subdirectory(operator)
add_subdirectory(pipeline)
add_subdirectory(tensor)

message(STATUS "use python11 build module")
pybind11_add_module(backend backend.cpp)
target_compile_definitions(backend PRIVATE VERSION_INFO=${ACCDATA_VERSION_INFO})
target_link_libraries(backend PRIVATE _accdata)
target_link_libraries(backend PRIVATE pthread)

install(TARGETS _accdata
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/output/AccData/lib)