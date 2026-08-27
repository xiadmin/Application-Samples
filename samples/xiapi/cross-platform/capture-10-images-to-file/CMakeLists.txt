cmake_minimum_required(VERSION 3.16)

project(xiapi-cross-platform-capture-10-images-to-file VERSION 0.1.0 LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../../../../libs/tiff-writer"
    "${CMAKE_CURRENT_BINARY_DIR}/libs/tiff-writer"
)

add_executable(capture_images_to_file_c main.c)
target_compile_features(capture_images_to_file_c PRIVATE c_std_11)
target_link_libraries(capture_images_to_file_c
    PRIVATE
        XIMEA::xiAPI
        ApplicationSamples::tiff_writer
)

sample_flat_output_directories(TARGET capture_images_to_file_c)

install(TARGETS capture_images_to_file_c RUNTIME DESTINATION bin)
