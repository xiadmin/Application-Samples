# Central list of external C/C++ libraries.
#
# One application_samples_dependency() record per library: where to get it, how
# to check it, the target it defines, and the options that configure its build.
# A sample asks for one with application_samples_use(<name>).
#
# The functions live in cmake/CMakeLists.txt. Include that file, not this one.

include_guard(GLOBAL)

if(NOT COMMAND application_samples_dependency)
    message(FATAL_ERROR
        "dependencies/Dependencies.cmake holds only the library list. Include "
        "cmake/CMakeLists.txt, which defines application_samples_dependency() "
        "and then includes this file.")
endif()

application_samples_dependency(
    NAME    libtiff
    URL     https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz
    SHA256  f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69
    TARGET  TIFF::tiff
    OPTIONS
        BUILD_SHARED_LIBS=OFF
        tiff-static=ON
        tiff-cxx=OFF
        tiff-tools=OFF
        tiff-tests=OFF
        tiff-contrib=OFF
        tiff-docs=OFF
        tiff-install=OFF
        jpeg=OFF
        old-jpeg=OFF
        jpeg12=OFF
        zlib=OFF
        libdeflate=OFF
        lzma=OFF
        zstd=OFF
        webp=OFF
        jbig=OFF
        lerc=OFF
        pixarlog=OFF
)

