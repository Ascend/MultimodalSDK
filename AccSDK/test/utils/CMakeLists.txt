set(LOG_TEST_EXECUTABLE "LogTest")
set(THREAD_POOL_TEST_EXECUTABLE "ThreadPoolTest")
set(FILE_UTILS_TEST_EXECUTABLE "FileUtilsTest")
set(VIDEO_UTILS_TEST_EXECUTABLE "VideoUtilsTest")
set(ERROR_UTILS_TEST_EXECUTABLE "ErrorCodeUtilsTest")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/utils)

file(GLOB_RECURSE LogTestSRC LogTest.cpp)
add_executable(${LOG_TEST_EXECUTABLE} ${LogTestSRC})
target_link_libraries(${LOG_TEST_EXECUTABLE} core -pthread gtest)
add_test(NAME ${LOG_TEST_EXECUTABLE}
        COMMAND ${LOG_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


file(GLOB_RECURSE ThreadPoolTestSRC ThreadPoolTest.cpp)
add_executable(${THREAD_POOL_TEST_EXECUTABLE} ${ThreadPoolTestSRC})
target_link_libraries(${THREAD_POOL_TEST_EXECUTABLE} core -pthread gtest)
add_test(NAME ${THREAD_POOL_TEST_EXECUTABLE}
        COMMAND ${THREAD_POOL_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB_RECURSE FileUtilTestSRC FileUtilsTest.cpp)
add_executable(${FILE_UTILS_TEST_EXECUTABLE} ${FileUtilTestSRC})
target_link_libraries(${FILE_UTILS_TEST_EXECUTABLE} core -pthread gtest dl)
add_test(NAME ${FILE_UTILS_TEST_EXECUTABLE}
        COMMAND ${FILE_UTILS_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB_RECURSE VideoUtilTestSRC VideoUtilsTest.cpp)
add_executable(${VIDEO_UTILS_TEST_EXECUTABLE} ${VideoUtilTestSRC})
target_link_libraries(${VIDEO_UTILS_TEST_EXECUTABLE} core -pthread gtest)
add_test(NAME ${VIDEO_UTILS_TEST_EXECUTABLE}
        COMMAND ${VIDEO_UTILS_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB_RECURSE ErrorCodeUtilTestSRC ErrorCodeUtilsTest.cpp)
add_executable(${ERROR_UTILS_TEST_EXECUTABLE} ${ErrorCodeUtilTestSRC})
target_link_libraries(${ERROR_UTILS_TEST_EXECUTABLE} core -pthread gtest)
add_test(NAME ${ERROR_UTILS_TEST_EXECUTABLE}
        COMMAND ${ERROR_UTILS_TEST_EXECUTABLE} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
