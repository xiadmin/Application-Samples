cmake_minimum_required(VERSION 3.16)

project(xiapiplus-cross-platform-demosaic-images-opencl VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# Without an OpenCL SDK the sample is skipped with a warning.
application_samples_find_opencl()
if(NOT APPLICATION_SAMPLES_OPENCL_FOUND)
    return()
endif()
application_samples_use(libtiff)

add_executable(demosaic_images_opencl
    main.cpp
    src/demosaic_kernel.cpp
    src/opencl_demosaic.cpp
    src/tiff_writer.cpp
)
target_compile_features(demosaic_images_opencl PRIVATE cxx_std_17)
target_include_directories(demosaic_images_opencl PRIVATE include)
# The sample uses the OpenCL 1.2 C API, which every current runtime provides.
target_compile_definitions(demosaic_images_opencl PRIVATE CL_TARGET_OPENCL_VERSION=120)
if(WIN32)
    # xiApi.h includes windows.h, whose min/max macros break std::min/std::max.
    target_compile_definitions(demosaic_images_opencl PRIVATE NOMINMAX)
endif()
target_link_libraries(demosaic_images_opencl
    PRIVATE
        XIMEA::xiAPIplus
        OpenCL::OpenCL
        TIFF::tiff
)

sample_flat_output_directories(TARGET demosaic_images_opencl)

install(TARGETS demosaic_images_opencl RUNTIME DESTINATION bin)
