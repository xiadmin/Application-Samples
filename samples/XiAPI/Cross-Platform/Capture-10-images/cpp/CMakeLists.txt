cmake_minimum_required(VERSION 3.16)

project(capture-10-images-cpp VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)

add_executable(capture_images_cpp main.cpp)
target_compile_features(capture_images_cpp PRIVATE cxx_std_17)
target_link_libraries(capture_images_cpp PRIVATE XIMEA::xiAPIplus)

sample_flat_output_directories(TARGET capture_images_cpp)

install(TARGETS capture_images_cpp RUNTIME DESTINATION bin)
