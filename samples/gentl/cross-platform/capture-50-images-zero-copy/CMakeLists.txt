cmake_minimum_required(VERSION 3.16)

project(gentl-cross-platform-capture-50-images-zero-copy VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# The GenTL headers and helpers ship with the XIMEA SP examples; without them
# the sample is skipped with a warning.
ximea_find_gentl()
if(NOT XIMEA_GENTL_FOUND)
    return()
endif()

add_executable(gentl_zero_copy main.cpp)
target_compile_features(gentl_zero_copy PRIVATE cxx_std_17)
target_link_libraries(gentl_zero_copy PRIVATE XIMEA::GenTLHelpers)

sample_flat_output_directories(TARGET gentl_zero_copy)

install(TARGETS gentl_zero_copy RUNTIME DESTINATION bin)
