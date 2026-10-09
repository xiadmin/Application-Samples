cmake_minimum_required(VERSION 3.16)

project(multirecorder-cross-platform-averaging-camera-client VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# The MultiRecorder sources ship with the XIMEA SP examples; without them the
# sample is skipped with a warning.
ximea_find_multirecorder()
if(NOT XIMEA_MULTIRECORDER_FOUND)
    return()
endif()

add_executable(averaging_camera_client
    main.cpp
    src/remote_camera_session.cpp
)
target_compile_features(averaging_camera_client PRIVATE cxx_std_17)
target_include_directories(averaging_camera_client PRIVATE include)
target_link_libraries(averaging_camera_client PRIVATE XIMEA::MultiRecorder)

sample_flat_output_directories(TARGET averaging_camera_client)

install(TARGETS averaging_camera_client RUNTIME DESTINATION bin)
