include_guard(GLOBAL)
include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Skip network update checks for already-downloaded FetchContent dependencies")

# application_samples_use_libtiff()
# Declares and makes available libtiff 4.7.1 as a minimal static library.
# After the call, TIFF::tiff is a valid link target.
function(application_samples_use_libtiff)
    if(TARGET TIFF::tiff)
        return()
    endif()

    # Minimal static build — no C++ binding, tools, tests, contrib, docs, install,
    # or optional compression dependencies.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(tiff-static        ON  CACHE BOOL "" FORCE)
    set(tiff-cxx           OFF CACHE BOOL "" FORCE)
    set(tiff-tools         OFF CACHE BOOL "" FORCE)
    set(tiff-tests         OFF CACHE BOOL "" FORCE)
    set(tiff-contrib       OFF CACHE BOOL "" FORCE)
    set(tiff-docs          OFF CACHE BOOL "" FORCE)
    set(tiff-install       OFF CACHE BOOL "" FORCE)
    set(jpeg               OFF CACHE BOOL "" FORCE)
    set(old-jpeg           OFF CACHE BOOL "" FORCE)
    set(jpeg12             OFF CACHE BOOL "" FORCE)
    set(zlib               OFF CACHE BOOL "" FORCE)
    set(libdeflate         OFF CACHE BOOL "" FORCE)
    set(lzma               OFF CACHE BOOL "" FORCE)
    set(zstd               OFF CACHE BOOL "" FORCE)
    set(webp               OFF CACHE BOOL "" FORCE)
    set(jbig               OFF CACHE BOOL "" FORCE)
    set(lerc               OFF CACHE BOOL "" FORCE)
    set(pixarlog           OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        libtiff
        URL      https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz
        URL_HASH SHA256=f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(libtiff)
endfunction()
