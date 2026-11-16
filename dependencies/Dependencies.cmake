# Central list of external C/C++ libraries.
#
# One application_samples_dependency() record per library: where to get it, the
# SHA256 of that archive, the target it defines, how to find an installed copy
# instead, and the options that configure its build. A sample asks for one with
# application_samples_use(<name>); the user decides with
# APPLICATION_SAMPLES_DEPENDENCIES whether it is downloaded or found installed.
#
# SHA256 is checked against every download and every cached archive. Take it
# from the archive you verified against the project's own signature or
# published checksum, never from a copy you have only downloaded.
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
    # Checked against tiff-4.7.1.tar.gz.sig, signed by libtiff maintainer
    # Even Rouault (key B1FA 7D81 EEB8 E663 9917 8B97 33EB BFC4 7B3D D87D).
    SHA256  f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69
    TARGET  TIFF::tiff
    # An installed libtiff, found with CMake's FindTIFF module.
    PACKAGE       TIFF
    SYSTEM_TARGET TIFF::TIFF
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
