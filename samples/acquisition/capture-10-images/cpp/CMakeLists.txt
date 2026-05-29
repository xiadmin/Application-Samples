cmake_minimum_required(VERSION 3.16)

project(capture-10-images-cpp VERSION 0.1.0 LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake")
include(SampleDefaults)
find_package(XIMEA REQUIRED)

add_executable(capture_images main.cpp)
target_compile_features(capture_images PRIVATE cxx_std_17)
target_link_libraries(capture_images PRIVATE XIMEA::xiAPIplus)

sample_flat_output_directories(TARGET capture_images)
