cmake_minimum_required(VERSION 3.16)

project(xiapi-cross-platform-capture-10-images VERSION 0.1.0 LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)

add_executable(xiapi_cross_platform_capture_10_images main.c)
target_compile_features(xiapi_cross_platform_capture_10_images PRIVATE c_std_11)
target_link_libraries(xiapi_cross_platform_capture_10_images PRIVATE XIMEA::xiAPI)

sample_flat_output_directories(TARGET xiapi_cross_platform_capture_10_images)

install(TARGETS xiapi_cross_platform_capture_10_images RUNTIME DESTINATION bin)
