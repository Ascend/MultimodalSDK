set(WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/output/py)

set(PY_TEST_SOURCE
        ${PROJECT_SOURCE_DIR}/py/accsdk_pytest.py
        ${PROJECT_SOURCE_DIR}/py/test_py_tensor.py
        ${PROJECT_SOURCE_DIR}/py/test_py_tensor_ops.py
        ${PROJECT_SOURCE_DIR}/py/test_py_preprocess.py
)

if(IMAGE)
    list(APPEND PY_TEST_SOURCE ${PROJECT_SOURCE_DIR}/py/test_py_image.py)
endif()
if(VIDEO)
    list(APPEND PY_TEST_SOURCE ${PROJECT_SOURCE_DIR}/py/test_py_video.py)
endif()

install(FILES ${PY_TEST_SOURCE} DESTINATION ${WORKING_DIRECTORY})

if(IMAGE)
    set(TEST_PY_IMAGE_EXECUTABLE "PyImageTest")
    add_test(NAME ${TEST_PY_IMAGE_EXECUTABLE}
            COMMAND python3 "test_py_image.py"
            WORKING_DIRECTORY ${WORKING_DIRECTORY})
    set_tests_properties(${TEST_PY_IMAGE_EXECUTABLE} PROPERTIES
            ENVIRONMENT PYTHONPATH=${PROJECT_SOURCE_DIR}/../output:$ENV{PYTHONPATH})
endif()

if(VIDEO)
    set(TEST_PY_VIDEO_EXECUTABLE "PyVideoTest")
    add_test(NAME ${TEST_PY_VIDEO_EXECUTABLE}
            COMMAND python3 "test_py_video.py"
            WORKING_DIRECTORY ${WORKING_DIRECTORY})
    set_tests_properties(${TEST_PY_VIDEO_EXECUTABLE} PROPERTIES
            ENVIRONMENT PYTHONPATH=${PROJECT_SOURCE_DIR}/../output:$ENV{PYTHONPATH})
endif()


set(TEST_PY_TENSOR_EXECUTABLE "PyTensorTest")
add_test(NAME ${TEST_PY_TENSOR_EXECUTABLE}
        COMMAND python3 "test_py_tensor.py"
        WORKING_DIRECTORY ${WORKING_DIRECTORY})
set_tests_properties(${TEST_PY_TENSOR_EXECUTABLE} PROPERTIES
        ENVIRONMENT PYTHONPATH=${PROJECT_SOURCE_DIR}/../output/:$ENV{PYTHONPATH})


set(TEST_PY_TENSOR_OPS_EXECUTABLE "PyTensorOpsTest")
add_test(NAME ${TEST_PY_TENSOR_OPS_EXECUTABLE}
        COMMAND python3 "test_py_tensor_ops.py"
        WORKING_DIRECTORY ${WORKING_DIRECTORY})
set_tests_properties(${TEST_PY_TENSOR_OPS_EXECUTABLE} PROPERTIES
        ENVIRONMENT PYTHONPATH=${PROJECT_SOURCE_DIR}/../output/:$ENV{PYTHONPATH})

set(TEST_PY_PREPROCESS_EXECUTABLE "PyPreprocessTest")
add_test(NAME ${TEST_PY_PREPROCESS_EXECUTABLE}
        COMMAND python3 "test_py_preprocess.py"
        WORKING_DIRECTORY ${WORKING_DIRECTORY})
set_tests_properties(${TEST_PY_PREPROCESS_EXECUTABLE} PROPERTIES
        ENVIRONMENT PYTHONPATH=${PROJECT_SOURCE_DIR}/../output/:$ENV{PYTHONPATH})