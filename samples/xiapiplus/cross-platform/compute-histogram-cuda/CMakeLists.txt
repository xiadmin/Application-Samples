# CMake 3.18 is the first version with find_package(CUDAToolkit) and
# CUDA_ARCHITECTURES, which the shared CUDA helper relies on.
cmake_minimum_required(VERSION 3.18)

project(xiapiplus-cross-platform-compute-histogram-cuda VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# Without the CUDA toolkit the sample is skipped with a warning.
application_samples_find_cuda()
if(NOT APPLICATION_SAMPLES_CUDA_FOUND)
    return()
endif()

add_executable(compute_histogram_cuda
    main.cpp
    src/cuda_helpers.cpp
    src/histogram64.cu
)
target_compile_features(compute_histogram_cuda PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(compute_histogram_cuda PRIVATE include)
if(WIN32)
    # xiApi.h includes windows.h, whose min/max macros break std::min/std::max.
    target_compile_definitions(compute_histogram_cuda PRIVATE NOMINMAX)
endif()
target_link_libraries(compute_histogram_cuda
    PRIVATE
        XIMEA::xiAPIplus
        CUDA::cudart
)

sample_flat_output_directories(TARGET compute_histogram_cuda)

install(TARGETS compute_histogram_cuda RUNTIME DESTINATION bin)
