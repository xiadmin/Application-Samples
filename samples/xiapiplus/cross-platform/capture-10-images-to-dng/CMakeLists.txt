cmake_minimum_required(VERSION 3.16)

project(xiapiplus-cross-platform-capture-10-images-to-dng VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# DNG Store ships with the XIMEA SP; without it the sample is skipped with a warning.
ximea_find_dng_store()
if(NOT XIMEA_DNG_STORE_FOUND)
    return()
endif()

add_executable(xiapiplus_cross_platform_capture_10_images_to_dng main.cpp)
target_compile_features(xiapiplus_cross_platform_capture_10_images_to_dng PRIVATE cxx_std_17)
target_link_libraries(xiapiplus_cross_platform_capture_10_images_to_dng
    PRIVATE
        XIMEA::xiAPIplus
        XIMEA::DngStore
)

sample_flat_output_directories(TARGET xiapiplus_cross_platform_capture_10_images_to_dng)
ximea_dng_store_runtime(TARGET xiapiplus_cross_platform_capture_10_images_to_dng)

install(TARGETS xiapiplus_cross_platform_capture_10_images_to_dng RUNTIME DESTINATION bin)
