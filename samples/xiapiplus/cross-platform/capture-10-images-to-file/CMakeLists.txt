cmake_minimum_required(VERSION 3.16)

project(xiapiplus-cross-platform-capture-10-images-to-file VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
# Request libtiff first: ximea_find_sdk() adds xiAPIplus_tiff.cpp to
# XIMEA::xiAPIplus only when the TIFF::tiff target already exists.
application_samples_use(libtiff)
ximea_find_sdk(REQUIRED)

add_executable(xiapiplus_cross_platform_capture_10_images_to_file main.cpp)
target_compile_features(xiapiplus_cross_platform_capture_10_images_to_file PRIVATE cxx_std_17)
target_link_libraries(xiapiplus_cross_platform_capture_10_images_to_file PRIVATE XIMEA::xiAPIplus)

sample_flat_output_directories(TARGET xiapiplus_cross_platform_capture_10_images_to_file)

install(TARGETS xiapiplus_cross_platform_capture_10_images_to_file RUNTIME DESTINATION bin)
