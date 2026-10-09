cmake_minimum_required(VERSION 3.16)

project(xiapiplus-cross-platform-capture-50-images-to-video-file-opencv VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# OpenCV is found with find_package(); without it, or without the xiAPIplus
# OpenCV wrapper from the XIMEA SP, the sample is skipped with a warning.
ximea_find_opencv_wrapper()
if(NOT XIMEA_OPENCV_FOUND)
    return()
endif()

add_executable(ocv_video_file main.cpp)
target_compile_features(ocv_video_file PRIVATE cxx_std_17)
target_link_libraries(ocv_video_file PRIVATE XIMEA::xiAPIplusOpenCV)

sample_flat_output_directories(TARGET ocv_video_file)

install(TARGETS ocv_video_file RUNTIME DESTINATION bin)
