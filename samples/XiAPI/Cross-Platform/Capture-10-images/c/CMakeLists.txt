cmake_minimum_required(VERSION 3.16)

project(capture-10-images-c VERSION 0.1.0 LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)

add_executable(capture_images_c main.c)
target_compile_features(capture_images_c PRIVATE c_std_11)
target_link_libraries(capture_images_c PRIVATE XIMEA::xiAPI)

sample_flat_output_directories(TARGET capture_images_c)

install(TARGETS capture_images_c RUNTIME DESTINATION bin)
