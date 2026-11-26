cmake_minimum_required(VERSION 3.16)

project(xiapi-hardware-specific-jetson-demosaic-sample VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_JETSON_DEMOSAIC_SAMPLE "Build the Nvidia Jetson demosaicing sample" ON)

if(NOT BUILD_JETSON_DEMOSAIC_SAMPLE)
    message(STATUS "Skipping Jetson demosaicing sample because BUILD_JETSON_DEMOSAIC_SAMPLE=OFF")
    return()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping Jetson demosaicing sample: Linux on Nvidia Jetson is required")
    return()
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(STATUS "Skipping Jetson demosaicing sample: Nvidia Jetson/aarch64 hardware is required")
    return()
endif()

enable_language(CUDA)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc cudaimgproc cudaarithm imgcodecs)

add_executable(xiapi_hardware_specific_jetson_demosaic_sample
    main.cpp
    depacking_cpu.cpp
    depacking.cu
)

target_compile_features(xiapi_hardware_specific_jetson_demosaic_sample PRIVATE cxx_std_17)
target_include_directories(xiapi_hardware_specific_jetson_demosaic_sample PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(xiapi_hardware_specific_jetson_demosaic_sample PRIVATE
    XIMEA::xiAPI
    CUDA::cudart
    CUDA::nppc
    CUDA::nppicc
    ${OpenCV_LIBS}
)

sample_flat_output_directories(TARGET xiapi_hardware_specific_jetson_demosaic_sample)

install(TARGETS xiapi_hardware_specific_jetson_demosaic_sample RUNTIME DESTINATION bin)
