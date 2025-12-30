set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/fusion_operators)

set(TEST_EXECUTABLE "FusionOperatorsTest")
file(GLOB_RECURSE COMPILE_TEST_SRC FusionOperatorsTest.cpp)
add_executable(${TEST_EXECUTABLE} ${COMPILE_TEST_SRC})
target_link_libraries(${TEST_EXECUTABLE} core securec -pthread gtest)
add_test(NAME ${TEST_EXECUTABLE}
        COMMAND ${TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})