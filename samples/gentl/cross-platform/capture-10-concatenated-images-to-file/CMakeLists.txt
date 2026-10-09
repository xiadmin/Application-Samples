cmake_minimum_required(VERSION 3.16)

project(gentl-cross-platform-capture-10-concatenated-images-to-file VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# The GenTL headers and helpers ship with the XIMEA SP examples; without them
# the sample is skipped with a warning.
ximea_find_gentl()
if(NOT XIMEA_GENTL_FOUND)
    return()
endif()
application_samples_use(libtiff)

add_executable(gentl_concatenated_images main.cpp)
target_compile_features(gentl_concatenated_images PRIVATE cxx_std_17)
target_link_libraries(gentl_concatenated_images
    PRIVATE
        XIMEA::GenTLHelpers
        TIFF::tiff
)

sample_flat_output_directories(TARGET gentl_concatenated_images)

install(TARGETS gentl_concatenated_images RUNTIME DESTINATION bin)
