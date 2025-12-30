set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/tensor)

# TensorOpsTest
set(TEST_EXECUTABLE "TensorOpsTest")
file(GLOB_RECURSE COMPILE_TEST_SRC TensorOpsTest.cpp)
add_executable(${TEST_EXECUTABLE} ${COMPILE_TEST_SRC})
target_link_libraries(${TEST_EXECUTABLE} core securec -pthread gtest)
add_test(NAME ${TEST_EXECUTABLE}
        COMMAND ${TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# TensorTest
set(TENSOR_TEST_EXECUTABLE "TensorTest")
file(GLOB_RECURSE TENSOR_TEST_SRC TensorTest.cpp)
add_executable(${TENSOR_TEST_EXECUTABLE} ${TENSOR_TEST_SRC})
target_link_libraries(${TENSOR_TEST_EXECUTABLE} core securec -pthread gtest)
add_test(NAME ${TENSOR_TEST_EXECUTABLE}
        COMMAND ${TENSOR_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# OpsCheckerTest
set(CHECKER_TEST_EXECUTABLE "OpsCheckerTest")
file(GLOB_RECURSE CHECKER_TEST_SRC OpsCheckerTest.cpp)
add_executable(${CHECKER_TEST_EXECUTABLE} ${CHECKER_TEST_SRC})
target_link_libraries(${CHECKER_TEST_EXECUTABLE} core securec -pthread gtest)
add_test(NAME ${CHECKER_TEST_EXECUTABLE}
        COMMAND ${CHECKER_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})