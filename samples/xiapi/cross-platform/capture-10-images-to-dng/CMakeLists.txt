cmake_minimum_required(VERSION 3.16)

project(xiapi-cross-platform-capture-10-images-to-dng VERSION 0.1.0 LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# DNG Store ships with the XIMEA SP; without it the sample is skipped with a warning.
ximea_find_dng_store()
if(NOT XIMEA_DNG_STORE_FOUND)
    return()
endif()

add_executable(xiapi_cross_platform_capture_10_images_to_dng main.c)
target_compile_features(xiapi_cross_platform_capture_10_images_to_dng PRIVATE c_std_11)
target_link_libraries(xiapi_cross_platform_capture_10_images_to_dng
    PRIVATE
        XIMEA::xiAPI
        XIMEA::DngStore
)

sample_flat_output_directories(TARGET xiapi_cross_platform_capture_10_images_to_dng)
ximea_dng_store_runtime(TARGET xiapi_cross_platform_capture_10_images_to_dng)

install(TARGETS xiapi_cross_platform_capture_10_images_to_dng RUNTIME DESTINATION bin)
