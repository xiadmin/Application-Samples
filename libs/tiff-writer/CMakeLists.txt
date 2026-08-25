add_library(application_samples_tiff_writer STATIC src/tiff_writer.c)
add_library(ApplicationSamples::tiff_writer ALIAS application_samples_tiff_writer)

target_compile_features(application_samples_tiff_writer PUBLIC c_std_11)
target_include_directories(application_samples_tiff_writer
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(application_samples_tiff_writer PUBLIC XIMEA::xiAPI)
