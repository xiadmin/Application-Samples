cmake_minimum_required(VERSION 3.16)

project(xiapi-cross-platform-offline-processing VERSION 0.1.0 LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CMakeLists.txt")
ximea_find_sdk(REQUIRED)
application_samples_use(libtiff)

add_executable(xiapi_cross_platform_offline_processing main.c)
target_compile_features(xiapi_cross_platform_offline_processing PRIVATE c_std_11)
target_link_libraries(xiapi_cross_platform_offline_processing
    PRIVATE
        XIMEA::xiAPI
        TIFF::tiff
)

sample_flat_output_directories(TARGET xiapi_cross_platform_offline_processing)

install(TARGETS xiapi_cross_platform_offline_processing RUNTIME DESTINATION bin)
