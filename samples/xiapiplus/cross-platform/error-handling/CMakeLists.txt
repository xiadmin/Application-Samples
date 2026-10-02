cmake_minimum_required(VERSION 3.16)

project(xiapiplus-cross-platform-error-handling VERSION 0.1.0 LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)

add_executable(xiapiplus_cross_platform_error_handling main.cpp)
target_compile_features(xiapiplus_cross_platform_error_handling PRIVATE cxx_std_17)
target_link_libraries(xiapiplus_cross_platform_error_handling PRIVATE XIMEA::xiAPIplus)

sample_flat_output_directories(TARGET xiapiplus_cross_platform_error_handling)

install(TARGETS xiapiplus_cross_platform_error_handling RUNTIME DESTINATION bin)
