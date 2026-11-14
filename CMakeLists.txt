cmake_minimum_required(VERSION 3.16)

project(PlanarParallaxMotionSegmentation
    VERSION 1.0.0
    DESCRIPTION "Three-frame planar-parallax motion detection and dense Graph Cut segmentation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build unit and integration tests" ON)
option(USE_OPENMP "Enable OpenMP multi-threading" ON)

set(CORE_SOURCES
    src/ImageSequenceReader.cpp
    src/ImageConverter.cpp
    src/GaussianPyramid.cpp
    src/ImageOps.cpp
    src/LinearSolver.cpp
    src/QuadraticFlowEstimator.cpp
    src/FrameAligner.cpp
    src/SecondaryFlowEstimator.cpp
    src/ParallaxRigidityTester.cpp
    src/DenseMotionSegmenter.cpp
    src/MotionDetectionPipeline.cpp
)

add_library(motion_seg_core STATIC ${CORE_SOURCES})

target_include_directories(motion_seg_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/header
)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(motion_seg_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP enabled for motion_seg_core")
    endif()
endif()

if(MSVC)
    target_compile_definitions(motion_seg_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
    target_compile_options(motion_seg_core PRIVATE /W4)
else()
    target_compile_options(motion_seg_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(motion_seg_cli src/main.cpp)
target_link_libraries(motion_seg_cli PRIVATE motion_seg_core)

if(MSVC)
    target_compile_options(motion_seg_cli PRIVATE /W4)
else()
    target_compile_options(motion_seg_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    enable_testing()

    set(TEST_FILES
        test/test_ImageConverter.cpp
        test/test_GaussianPyramid.cpp
        test/test_ImageOps.cpp
        test/test_LinearSolver.cpp
        test/test_QuadraticFlowEstimator.cpp
        test/test_FrameAligner.cpp
        test/test_SecondaryFlowEstimator.cpp
        test/test_ParallaxRigidityTester.cpp
        test/test_DenseMotionSegmenter.cpp
        test/test_MotionDetectionPipeline.cpp
    )

    foreach(TEST_REL_PATH IN LISTS TEST_FILES)
        set(TEST_ABS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_REL_PATH}")
        if(EXISTS "${TEST_ABS_PATH}")
            get_filename_component(TEST_NAME "${TEST_REL_PATH}" NAME_WE)
            add_executable("${TEST_NAME}" "${TEST_ABS_PATH}")
            target_link_libraries("${TEST_NAME}" PRIVATE motion_seg_core)
            add_test(
                NAME "${TEST_NAME}"
                COMMAND "${TEST_NAME}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            )
        endif()
    endforeach()
endif()