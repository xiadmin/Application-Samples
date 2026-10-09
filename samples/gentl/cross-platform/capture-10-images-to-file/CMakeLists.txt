cmake_minimum_required(VERSION 3.16)

project(gentl-cross-platform-capture-10-images-to-file VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
# The GenTL headers ship with the XIMEA SP examples; without them the sample is
# skipped with a warning.
ximea_find_gentl()
if(NOT XIMEA_GENTL_FOUND)
    return()
endif()
application_samples_use(libtiff)

add_executable(gentl_capture_to_file
    main.cpp
    src/gentl_producer.cpp
    src/tiff_writer.cpp
)
target_compile_features(gentl_capture_to_file PRIVATE cxx_std_17)
target_include_directories(gentl_capture_to_file PRIVATE include)
# The producer is loaded at run time, so the sample links no XIMEA library.
target_link_libraries(gentl_capture_to_file
    PRIVATE
        XIMEA::GenTL
        TIFF::tiff
)

sample_flat_output_directories(TARGET gentl_capture_to_file)

install(TARGETS gentl_capture_to_file RUNTIME DESTINATION bin)
