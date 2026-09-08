include_guard(GLOBAL)
include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Skip network update checks for already-downloaded FetchContent dependencies")

# application_samples_dependency(NAME <name> URL <url> SHA256 <hash> TARGET <target>
#                                [OPTIONS <variable>=<value>...])
# Records one external dependency. Nothing is downloaded here; the record is read
# by application_samples_use(). OPTIONS are cache variables set before the fetch,
# so they control how the dependency itself is configured.
function(application_samples_dependency)
    cmake_parse_arguments(ARG "" "NAME;URL;SHA256;TARGET" "OPTIONS" ${ARGN})

    foreach(required IN ITEMS NAME URL SHA256 TARGET)
        if(NOT ARG_${required})
            message(FATAL_ERROR
                "application_samples_dependency: ${required} is required")
        endif()
    endforeach()
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "application_samples_dependency(${ARG_NAME}): unexpected arguments: "
            "${ARG_UNPARSED_ARGUMENTS}")
    endif()

    get_property(_known GLOBAL PROPERTY APPLICATION_SAMPLES_DEPENDENCIES)
    if("${ARG_NAME}" IN_LIST _known)
        message(FATAL_ERROR
            "application_samples_dependency(${ARG_NAME}): declared twice. One "
            "library must have exactly one central version.")
    endif()

    set_property(GLOBAL APPEND PROPERTY APPLICATION_SAMPLES_DEPENDENCIES "${ARG_NAME}")
    foreach(field IN ITEMS URL SHA256 TARGET OPTIONS)
        set_property(GLOBAL PROPERTY
            APPLICATION_SAMPLES_DEPENDENCY_${ARG_NAME}_${field} "${ARG_${field}}")
    endforeach()
endfunction()

# application_samples_use(<name>)
# Downloads and builds a recorded dependency, then makes its target available.
# A sample calls this; a sample never selects the version.
function(application_samples_use name)
    get_property(_known GLOBAL PROPERTY APPLICATION_SAMPLES_DEPENDENCIES)
    if(NOT "${name}" IN_LIST _known)
        message(FATAL_ERROR
            "application_samples_use(${name}): no such central dependency. "
            "Declared: ${_known}. Add it to dependencies/Dependencies.cmake.")
    endif()

    foreach(field IN ITEMS URL SHA256 TARGET OPTIONS)
        get_property(_${field} GLOBAL PROPERTY
            APPLICATION_SAMPLES_DEPENDENCY_${name}_${field})
    endforeach()

    if(TARGET ${_TARGET})
        return()
    endif()

    # Cache entries are global, so these reach the dependency's own CMake code.
    foreach(entry IN LISTS _OPTIONS)
        if(NOT entry MATCHES "^([^=]+)=(.*)$")
            message(FATAL_ERROR
                "application_samples_use(${name}): option '${entry}' is not "
                "<variable>=<value>")
        endif()
        set(variable "${CMAKE_MATCH_1}")
        set(value "${CMAKE_MATCH_2}")
        string(TOUPPER "${value}" _upper_value)
        if(_upper_value MATCHES "^(ON|OFF|TRUE|FALSE|YES|NO|0|1)$")
            set(${variable} ${value} CACHE BOOL "" FORCE)
        else()
            set(${variable} ${value} CACHE STRING "" FORCE)
        endif()
    endforeach()

    FetchContent_Declare(
        ${name}
        URL      ${_URL}
        URL_HASH SHA256=${_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(${name})

    if(NOT TARGET ${_TARGET})
        message(FATAL_ERROR
            "application_samples_use(${name}): built, but target ${_TARGET} was "
            "not defined. Correct TARGET in dependencies/Dependencies.cmake.")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Central dependency records
# ---------------------------------------------------------------------------

# Minimal static build — no C++ binding, tools, tests, contrib, docs, install,
# or optional compression dependencies.
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

# application_samples_use_libtiff()
# Deprecated spelling kept for existing samples; use application_samples_use(libtiff).
function(application_samples_use_libtiff)
    application_samples_use(libtiff)
endfunction()
