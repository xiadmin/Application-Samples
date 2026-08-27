cmake_minimum_required(VERSION 3.16)

project(xiapi-cross-platform-capture-10-images-to-ram VERSION 0.1.0 LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)

add_executable(capture_images_to_ram_c main.c)
target_compile_features(capture_images_to_ram_c PRIVATE c_std_11)
target_link_libraries(capture_images_to_ram_c PRIVATE XIMEA::xiAPI)

sample_flat_output_directories(TARGET capture_images_to_ram_c)

install(TARGETS capture_images_to_ram_c RUNTIME DESTINATION bin)
