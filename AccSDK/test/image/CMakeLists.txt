set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/image)

# ImageTest
set(TEST_EXECUTABLE "ImageTest")
file(GLOB_RECURSE COMPILE_TEST_SRC ImageTest.cpp)
add_executable(${TEST_EXECUTABLE} ${COMPILE_TEST_SRC})
target_link_libraries(${TEST_EXECUTABLE} core -pthread gtest)

add_test(NAME ${TEST_EXECUTABLE}
        COMMAND ${TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ImageOpsTest
set(IMAGE_OP_TEST_EXECUTABLE "ImageOpsTest")
file(GLOB_RECURSE IMAGE_OP_TEST_SRC ImageOpsTest.cpp)
add_executable(${IMAGE_OP_TEST_EXECUTABLE} ${IMAGE_OP_TEST_SRC})
target_link_libraries(${IMAGE_OP_TEST_EXECUTABLE} core -pthread gtest)

add_test(NAME ${IMAGE_OP_TEST_EXECUTABLE}
        COMMAND ${IMAGE_OP_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})