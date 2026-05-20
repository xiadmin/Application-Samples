cmake_minimum_required(VERSION 3.16)

project(capture-10-images-c VERSION 0.1.0 LANGUAGES C)

# Include shared
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake")
include(SampleDefaults)
find_package(XIMEA REQUIRED)

add_executable(capture_images main.c)
target_compile_features(capture_images PRIVATE c_std_11)
target_link_libraries(capture_images PRIVATE XIMEA::xiAPI)

# Apply output directory flattening to this target
sample_flat_output_directories(TARGET capture_images)