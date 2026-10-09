cmake_minimum_required(VERSION 3.16)

project(xiapiplus-cross-platform-capture-25-images-in-parallel VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
find_package(Threads REQUIRED)

add_executable(capture_in_parallel_cpp
    main.cpp
    src/frame_queue.cpp
    src/synchronized_camera.cpp
)
target_compile_features(capture_in_parallel_cpp PRIVATE cxx_std_17)
target_include_directories(capture_in_parallel_cpp PRIVATE include)
target_link_libraries(capture_in_parallel_cpp
    PRIVATE
        XIMEA::xiAPIplus
        Threads::Threads
)

sample_flat_output_directories(TARGET capture_in_parallel_cpp)

install(TARGETS capture_in_parallel_cpp RUNTIME DESTINATION bin)
