cmake_minimum_required(VERSION 3.16)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
    project(ximea-sample-cmake-helpers LANGUAGES C CXX)
endif()

include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Skip network update checks for already-downloaded FetchContent dependencies")

# Archive cache shared by every sample build tree, so a library is downloaded
# once per machine instead of once per sample. Archive contents are not
# verified; a version change is picked up through the URL in the central list.
get_filename_component(_application_samples_cache_default
    "${CMAKE_CURRENT_LIST_DIR}/../.deps" ABSOLUTE)
set(APPLICATION_SAMPLES_CACHE_DIR "${_application_samples_cache_default}" CACHE PATH
    "Shared directory holding downloaded dependency archives")
unset(_application_samples_cache_default)

option(APPLICATION_SAMPLES_OFFLINE
    "Report a missing cached archive instead of reaching for the network" OFF)

function(application_samples_dependency)
    cmake_parse_arguments(ARG "" "NAME;URL;TARGET" "OPTIONS" ${ARGN})

    foreach(required IN ITEMS NAME URL TARGET)
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
    foreach(field IN ITEMS URL TARGET OPTIONS)
        set_property(GLOBAL PROPERTY
            APPLICATION_SAMPLES_DEPENDENCY_${ARG_NAME}_${field} "${ARG_${field}}")
    endforeach()
endfunction()

# Downloads and builds a recorded dependency, then makes its target available.
function(application_samples_use name)
    get_property(_known GLOBAL PROPERTY APPLICATION_SAMPLES_DEPENDENCIES)
    if(NOT "${name}" IN_LIST _known)
        message(FATAL_ERROR
            "application_samples_use(${name}): no such central dependency. "
            "Declared: ${_known}. Add it to dependencies/Dependencies.cmake.")
    endif()

    foreach(field IN ITEMS URL TARGET OPTIONS)
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

    set(_download_dir "${APPLICATION_SAMPLES_CACHE_DIR}/downloads")
    get_filename_component(_archive "${_URL}" NAME)

    set(_hash_args)
    if(EXISTS "${_download_dir}/${_archive}")
        # The download step keeps an archive only when it can match it against a
        # hash, and deletes it otherwise. Hand it the hash of the copy already on
        # disk so the cache survives; this checks nothing about the contents.
        file(SHA256 "${_download_dir}/${_archive}" _cached_sha256)
        set(_hash_args URL_HASH SHA256=${_cached_sha256})
        message(STATUS
            "application_samples_use(${name}): reusing cached ${_archive}")
    elseif(APPLICATION_SAMPLES_OFFLINE)
        # A failed download is retried five times with growing sleeps, so an
        # offline build would otherwise stall for minutes before failing.
        message(FATAL_ERROR
            "application_samples_use(${name}): ${_archive} is not in the cache "
            "at ${_download_dir}, and APPLICATION_SAMPLES_OFFLINE is ON. "
            "Configure once with network access, or copy ${_URL} into that "
            "directory.")
    else()
        message(STATUS
            "application_samples_use(${name}): downloading ${_archive} into "
            "${_download_dir}; later builds reuse it and need no network")
    endif()

    FetchContent_Declare(
        ${name}
        URL                ${_URL}
        ${_hash_args}
        DOWNLOAD_DIR       "${_download_dir}"
        TIMEOUT            600
        INACTIVITY_TIMEOUT 30
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(${name})

    if(NOT TARGET ${_TARGET})
        message(FATAL_ERROR
            "application_samples_use(${name}): built, but target ${_TARGET} was "
            "not defined. Correct TARGET in dependencies/Dependencies.cmake.")
    endif()
endfunction()

# The central library list. It calls application_samples_dependency() above,
# so it is included after these functions exist.
include("${CMAKE_CURRENT_LIST_DIR}/../dependencies/Dependencies.cmake")

function(_sample_derive_output_name out_var)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}")
    string(REPLACE "\\" "/" _src "${_src}")

    string(REGEX MATCH ".*/[Ss]amples/(.+)$" _match "${_src}")
    if(_match)
        set(_rel "${CMAKE_MATCH_1}")
        string(REPLACE "/" "-" _name "${_rel}")
    else()
        get_filename_component(_name "${_src}" NAME)
    endif()

    set(${out_var} "${_name}" PARENT_SCOPE)
endfunction()

function(sample_flat_output_directories)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "sample_flat_output_directories: TARGET must be specified")
    endif()

    _sample_derive_output_name(_output_name)

    set_target_properties(${ARG_TARGET} PROPERTIES
        OUTPUT_NAME "${_output_name}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/build"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build"
    )
endfunction()

function(ximea_find_sdk)
    set(options REQUIRED)
    set(oneValueArgs)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(POLICY CMP0074)
        cmake_policy(SET CMP0074 NEW)
    endif()

    set(_ximea_sp_path "$ENV{XIMEA_SP_PATH}")
    set(_ximea_root_hints "${_ximea_sp_path}" "${XIMEA_ROOT}" "$ENV{XIMEA_ROOT}")
    set(_ximea_root_inc_hints)
    set(_ximea_root_lib_hints)
    set(_ximea_root_win_inc_hints)
    set(_ximea_root_win_inc_plus_hints)
    set(_ximea_root_win_lib_hints)
    foreach(_ximea_root IN LISTS _ximea_root_hints)
        if(_ximea_root)
            list(APPEND _ximea_root_inc_hints "${_ximea_root}/include")
            list(APPEND _ximea_root_lib_hints "${_ximea_root}/lib" "${_ximea_root}/lib64")
            list(APPEND _ximea_root_win_inc_hints "${_ximea_root}/API/xiAPI")
            list(APPEND _ximea_root_win_inc_plus_hints "${_ximea_root}/Examples/Sources/_libs/xiAPIplus")
            list(APPEND _ximea_root_win_lib_hints "${_ximea_root}/API/xiAPI")
        endif()
    endforeach()

    if(WIN32)
        set(_ximea_inc_hints "${_ximea_sp_path}/API/xiAPI" ${_ximea_root_win_inc_hints})
        set(_ximea_inc_plus_hints "${_ximea_sp_path}/Examples/Sources/_libs/xiAPIplus" ${_ximea_root_win_inc_plus_hints})
        set(_ximea_lib_hints "${_ximea_sp_path}/API/xiAPI" ${_ximea_root_win_lib_hints})
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(_ximea_lib_names xiapi64)
        else()
            set(_ximea_lib_names xiapi32)
        endif()
    elseif(APPLE)
        set(_ximea_inc_hints ${_ximea_root_inc_hints} "/Library/Frameworks/m3api.framework/Headers")
        set(_ximea_inc_plus_hints "${_ximea_sp_path}/Examples/Sources/_libs/xiAPIplus" ${_ximea_root_inc_hints} "/Library/Frameworks/m3api.framework/Headers")
        set(_ximea_lib_hints ${_ximea_root_lib_hints} "/Library/Frameworks/m3api.framework")
        set(_ximea_lib_names m3api)
    else()
        set(_ximea_inc_hints ${_ximea_root_inc_hints} "/opt/XIMEA/include")
        set(_ximea_inc_plus_hints ${_ximea_root_inc_hints} "/opt/XIMEA/samples/_libs/xiAPIplus" "/opt/XIMEA/include")
        set(_ximea_lib_hints ${_ximea_root_lib_hints} "/opt/XIMEA/lib" "/opt/XIMEA/lib64")
        set(_ximea_lib_names m3api)
    endif()

    find_path(XIMEA_INCLUDE_DIR
        NAMES xiApi.h
        HINTS ${_ximea_inc_hints}
    )

    find_path(XIMEA_INCLUDE_PLUS_DIR
        NAMES xiapiplus.h xiApiPlus.h
        HINTS ${_ximea_inc_plus_hints}
    )

    find_library(XIMEA_LIBRARY
        NAMES ${_ximea_lib_names}
        HINTS ${_ximea_lib_hints}
    )

    include(FindPackageHandleStandardArgs)
    set(_ximea_required_vars XIMEA_LIBRARY XIMEA_INCLUDE_DIR)
    if(CMAKE_CXX_COMPILER_LOADED)
        list(APPEND _ximea_required_vars XIMEA_INCLUDE_PLUS_DIR)
    endif()
    find_package_handle_standard_args(XIMEA
        REQUIRED_VARS ${_ximea_required_vars}
    )

    if(NOT XIMEA_FOUND)
        message(WARNING
            "XIMEA SDK not found. Install the XIMEA Software Package and set "
            "XIMEA_ROOT (cmake -DXIMEA_ROOT=<path>) or the XIMEA_SP_PATH environment variable. "
            "xiapiplus.h is expected at <XIMEA_SP_PATH>/Examples/Sources/_libs/xiAPIplus "
            "on Windows, /opt/XIMEA/samples/_libs/xiAPIplus on Linux, or set XIMEA_INCLUDE_PLUS_DIR manually.")
        if(ARG_REQUIRED)
            message(FATAL_ERROR "Required XIMEA SDK was not found")
        endif()
    endif()

    if(XIMEA_FOUND AND NOT TARGET XIMEA::xiAPI)
        add_library(XIMEA::xiAPI UNKNOWN IMPORTED)
        set_target_properties(XIMEA::xiAPI PROPERTIES
            IMPORTED_LOCATION "${XIMEA_LIBRARY}"
        )
        target_include_directories(XIMEA::xiAPI INTERFACE "${XIMEA_INCLUDE_DIR}")
    endif()

    if(XIMEA_FOUND AND CMAKE_CXX_COMPILER_LOADED AND NOT TARGET XIMEA::xiAPIplus)
        if(EXISTS "${XIMEA_INCLUDE_PLUS_DIR}/xiAPIplus_core.cpp")
            # libtiff names its own target TIFF::tiff, CMake's FindTIFF module
            # names the system one TIFF::TIFF. Prefer the central dependency
            # when the sample asked for it with application_samples_use(libtiff)
            # before calling this function.
            set(_ximea_tiff_target "")
            if(TARGET TIFF::tiff)
                set(_ximea_tiff_target TIFF::tiff)
            else()
                find_package(TIFF QUIET)
                if(TIFF_FOUND)
                    set(_ximea_tiff_target TIFF::TIFF)
                endif()
            endif()

            set(_ximea_plus_sources "${XIMEA_INCLUDE_PLUS_DIR}/xiAPIplus_core.cpp")
            if(_ximea_tiff_target)
                list(APPEND _ximea_plus_sources "${XIMEA_INCLUDE_PLUS_DIR}/xiAPIplus_tiff.cpp")
            endif()

            if(NOT TARGET XIMEA_xiAPIplus)
                add_library(XIMEA_xiAPIplus STATIC ${_ximea_plus_sources})
                set_target_properties(XIMEA_xiAPIplus PROPERTIES LINKER_LANGUAGE CXX)
                get_filename_component(_ximea_libs_dir "${XIMEA_INCLUDE_PLUS_DIR}" DIRECTORY)
                target_include_directories(XIMEA_xiAPIplus PUBLIC
                    "${XIMEA_INCLUDE_PLUS_DIR}"
                    "${_ximea_libs_dir}"
                    "${_ximea_libs_dir}/libtiff"
                )
                if(NOT WIN32)
                    target_compile_definitions(XIMEA_xiAPIplus PRIVATE UNIX)
                endif()
                target_link_libraries(XIMEA_xiAPIplus PUBLIC XIMEA::xiAPI)

                if(_ximea_tiff_target)
                    target_link_libraries(XIMEA_xiAPIplus PUBLIC ${_ximea_tiff_target})
                    # XIMEA SP 4.33 xiAPIplus_tiff.cpp omits <cstring>.
                    set(_tiff_src "${XIMEA_INCLUDE_PLUS_DIR}/xiAPIplus_tiff.cpp")
                    if(MSVC)
                        set_source_files_properties("${_tiff_src}" PROPERTIES
                            COMPILE_OPTIONS "/FIcstring")
                    else()
                        set_source_files_properties("${_tiff_src}" PROPERTIES
                            COMPILE_OPTIONS "-include;cstring")
                    endif()
                endif()
            endif()

            add_library(XIMEA::xiAPIplus ALIAS XIMEA_xiAPIplus)
        else()
            add_library(XIMEA::xiAPIplus INTERFACE IMPORTED)
            target_include_directories(XIMEA::xiAPIplus INTERFACE "${XIMEA_INCLUDE_PLUS_DIR}")
            target_link_libraries(XIMEA::xiAPIplus INTERFACE XIMEA::xiAPI)
        endif()
    endif()

    mark_as_advanced(XIMEA_INCLUDE_DIR XIMEA_INCLUDE_PLUS_DIR XIMEA_LIBRARY)
endfunction()

# Records in the build tree why a sample skipped itself, so build tooling can
# report it as skipped instead of as a build without output. Called by the
# ximea_find_* helpers below; a found component removes a stale marker.
function(_ximea_sample_skip_marker)
    cmake_parse_arguments(ARG "" "REASON" "" ${ARGN})
    set(_marker "${CMAKE_BINARY_DIR}/sample-skipped.txt")
    if(ARG_REASON)
        file(WRITE "${_marker}" "${ARG_REASON}
")
    else()
        file(REMOVE "${_marker}")
    endif()
endfunction()

# Finds the DNG Store library that ships with the XIMEA Software Package and
# defines XIMEA::DngStore. It is part of the SDK like xiAPI, so it carries no
# central version. Call ximea_find_sdk() first; a sample checks
# XIMEA_DNG_STORE_FOUND and skips itself when the SP component is missing.
function(ximea_find_dng_store)
    set(_ximea_sp_path "$ENV{XIMEA_SP_PATH}")
    set(_dng_roots "${_ximea_sp_path}" "${XIMEA_ROOT}" "$ENV{XIMEA_ROOT}")
    set(_dng_inc_hints)
    set(_dng_lib_hints)
    foreach(_root IN LISTS _dng_roots)
        if(_root)
            list(APPEND _dng_inc_hints "${_root}/DNG Store/Include" "${_root}/include")
            list(APPEND _dng_lib_hints "${_root}/DNG Store/binX64" "${_root}/lib" "${_root}/lib64")
        endif()
    endforeach()
    if(APPLE)
        list(APPEND _dng_inc_hints "/Library/Frameworks/m3api.framework/Headers")
        list(APPEND _dng_lib_hints "/Library/Frameworks/m3api.framework/Libraries"
            "/Library/Frameworks/m3api.framework/Versions/B/Libraries")
    elseif(NOT WIN32)
        # The Linux installer copies the header into /opt/XIMEA/include, links
        # /usr/include/m3api to it, and puts the library in /opt/XIMEA/lib.
        list(APPEND _dng_inc_hints "/opt/XIMEA/include")
        list(APPEND _dng_lib_hints "/opt/XIMEA/lib" "/opt/XIMEA/lib64")
    endif()

    find_path(XIMEA_DNG_STORE_INCLUDE_DIR NAMES xiapi_dng_store.h HINTS ${_dng_inc_hints})
    find_library(XIMEA_DNG_STORE_LIBRARY NAMES xiapi_dng_store HINTS ${_dng_lib_hints})
    if(WIN32)
        find_file(XIMEA_DNG_STORE_DLL NAMES xiapi_dng_store.dll HINTS ${_dng_lib_hints})
    endif()
    mark_as_advanced(XIMEA_DNG_STORE_INCLUDE_DIR XIMEA_DNG_STORE_LIBRARY XIMEA_DNG_STORE_DLL)

    set(_dng_found FALSE)
    if(TARGET XIMEA::xiAPI AND XIMEA_DNG_STORE_INCLUDE_DIR AND XIMEA_DNG_STORE_LIBRARY
       AND (NOT WIN32 OR XIMEA_DNG_STORE_DLL))
        set(_dng_found TRUE)
    endif()
    set(XIMEA_DNG_STORE_FOUND ${_dng_found} PARENT_SCOPE)

    if(NOT _dng_found)
        message(WARNING
            "XIMEA DNG Store not found, skipping ${PROJECT_NAME}. Install the "
            "\"DNG Store\" component of the XIMEA Software Package; on Windows it "
            "is expected at <XIMEA_SP_PATH>/DNG Store/{Include,binX64}.")
        _ximea_sample_skip_marker(REASON "XIMEA DNG Store not found")
        return()
    endif()
    _ximea_sample_skip_marker()

    if(NOT TARGET XIMEA::DngStore)
        if(WIN32)
            add_library(XIMEA::DngStore SHARED IMPORTED GLOBAL)
            set_target_properties(XIMEA::DngStore PROPERTIES
                IMPORTED_LOCATION "${XIMEA_DNG_STORE_DLL}"
                IMPORTED_IMPLIB "${XIMEA_DNG_STORE_LIBRARY}")
        else()
            add_library(XIMEA::DngStore UNKNOWN IMPORTED GLOBAL)
            set_target_properties(XIMEA::DngStore PROPERTIES
                IMPORTED_LOCATION "${XIMEA_DNG_STORE_LIBRARY}")
            if(APPLE)
                # xiapi_dng_store.h includes <m3api/xiApi.h>, a framework header.
                target_compile_options(XIMEA::DngStore INTERFACE "-F/Library/Frameworks")
            endif()
        endif()
        # Linked after XIMEA::xiAPI, so the SDK's own xiApi.h stays first on the
        # include path instead of the copy in DNG Store/Include.
        target_include_directories(XIMEA::DngStore INTERFACE "${XIMEA_DNG_STORE_INCLUDE_DIR}")
        target_link_libraries(XIMEA::DngStore INTERFACE XIMEA::xiAPI)
    endif()
endfunction()

# xiapi_dng_store.dll is not on PATH after an SP install, so put it next to the
# executable in the build tree and in the install prefix.
function(ximea_dng_store_runtime)
    cmake_parse_arguments(ARG "" "TARGET" "" ${ARGN})
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "ximea_dng_store_runtime: TARGET must be specified")
    endif()
    if(WIN32 AND XIMEA_DNG_STORE_DLL)
        add_custom_command(TARGET ${ARG_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${XIMEA_DNG_STORE_DLL}" "$<TARGET_FILE_DIR:${ARG_TARGET}>")
        install(FILES "${XIMEA_DNG_STORE_DLL}" DESTINATION bin)
    endif()
endfunction()

# Finds the GenTL headers and consumer helpers that ship with the XIMEA Software
# Package under Examples/Sources/_libs/GenTL and defines:
#   XIMEA::GenTL         headers only, for consumers that load a GenTL producer
#                        (.cti) at run time; adds the dlopen library off Windows.
#   XIMEA::GenTLHelpers  xiGenTL/GenTLStream/ImgBuffer helpers built from the SP
#                        sources. They call the GenTL functions that xiAPI
#                        exports, so they link XIMEA::xiAPI.
# Call ximea_find_sdk() first; a sample checks XIMEA_GENTL_FOUND and skips
# itself when the SP component is missing.
function(ximea_find_gentl)
    set(_ximea_sp_path "$ENV{XIMEA_SP_PATH}")
    set(_gentl_roots "${_ximea_sp_path}" "${XIMEA_ROOT}" "$ENV{XIMEA_ROOT}")
    set(_gentl_inc_hints)
    foreach(_root IN LISTS _gentl_roots)
        if(_root)
            list(APPEND _gentl_inc_hints
                "${_root}/Examples/Sources/_libs/GenTL"
                "${_root}/samples/_libs/GenTL")
        endif()
    endforeach()
    if(NOT WIN32)
        list(APPEND _gentl_inc_hints "/opt/XIMEA/samples/_libs/GenTL")
    endif()

    find_path(XIMEA_GENTL_INCLUDE_DIR NAMES GenTL_v1_5.h HINTS ${_gentl_inc_hints})
    mark_as_advanced(XIMEA_GENTL_INCLUDE_DIR)

    set(_gentl_found FALSE)
    if(TARGET XIMEA::xiAPI AND XIMEA_GENTL_INCLUDE_DIR)
        set(_gentl_found TRUE)
    endif()
    set(XIMEA_GENTL_FOUND ${_gentl_found} PARENT_SCOPE)

    if(NOT _gentl_found)
        message(WARNING
            "XIMEA GenTL headers not found, skipping ${PROJECT_NAME}. Install the "
            "XIMEA Software Package with its examples; GenTL_v1_5.h is expected at "
            "<XIMEA_SP_PATH>/Examples/Sources/_libs/GenTL on Windows or "
            "/opt/XIMEA/samples/_libs/GenTL on Linux.")
        _ximea_sample_skip_marker(REASON "XIMEA GenTL headers not found")
        return()
    endif()
    _ximea_sample_skip_marker()

    if(NOT TARGET XIMEA::GenTL)
        add_library(XIMEA::GenTL INTERFACE IMPORTED GLOBAL)
        target_include_directories(XIMEA::GenTL INTERFACE "${XIMEA_GENTL_INCLUDE_DIR}")
        if(NOT WIN32)
            target_link_libraries(XIMEA::GenTL INTERFACE ${CMAKE_DL_LIBS})
        endif()
    endif()

    if(NOT TARGET XIMEA_GenTLHelpers)
        add_library(XIMEA_GenTLHelpers STATIC EXCLUDE_FROM_ALL
            "${XIMEA_GENTL_INCLUDE_DIR}/GenTLStream.cpp"
            "${XIMEA_GENTL_INCLUDE_DIR}/xi_GenTL.cpp"
            "${XIMEA_GENTL_INCLUDE_DIR}/xi_ImgBuffer.cpp"
            "${XIMEA_GENTL_INCLUDE_DIR}/xi_exception.cpp"
        )
        set_target_properties(XIMEA_GenTLHelpers PROPERTIES LINKER_LANGUAGE CXX)
        target_compile_features(XIMEA_GenTLHelpers PUBLIC cxx_std_17)
        if(NOT WIN32)
            target_compile_definitions(XIMEA_GenTLHelpers PRIVATE UNIX)
        endif()
        target_link_libraries(XIMEA_GenTLHelpers PUBLIC XIMEA::GenTL XIMEA::xiAPI)
        add_library(XIMEA::GenTLHelpers ALIAS XIMEA_GenTLHelpers)
    endif()
endfunction()

# Finds OpenCV with find_package() and builds the xiAPIplus OpenCV wrapper that
# ships with the XIMEA Software Package (_libs/xiAPIplusOpenCV) as
# XIMEA::xiAPIplusOpenCV. xiApiPlusOcv.hpp carries its own copy of the
# xiAPIplus classes, so the target links XIMEA::xiAPI and must not be combined
# with XIMEA::xiAPIplus. Call ximea_find_sdk() first; a sample checks
# XIMEA_OPENCV_FOUND and skips itself when OpenCV or the wrapper is missing.
function(ximea_find_opencv_wrapper)
    set(_ximea_sp_path "$ENV{XIMEA_SP_PATH}")
    set(_ocv_roots "${_ximea_sp_path}" "${XIMEA_ROOT}" "$ENV{XIMEA_ROOT}")
    set(_ocv_inc_hints)
    foreach(_root IN LISTS _ocv_roots)
        if(_root)
            list(APPEND _ocv_inc_hints
                "${_root}/Examples/Sources/_libs/xiAPIplusOpenCV"
                "${_root}/samples/_libs/xiAPIplusOpenCV")
        endif()
    endforeach()
    if(NOT WIN32)
        list(APPEND _ocv_inc_hints "/opt/XIMEA/samples/_libs/xiAPIplusOpenCV")
    endif()

    find_path(XIMEA_OPENCV_WRAPPER_DIR NAMES xiApiPlusOcv.hpp HINTS ${_ocv_inc_hints})
    mark_as_advanced(XIMEA_OPENCV_WRAPPER_DIR)
    find_package(OpenCV QUIET COMPONENTS core imgproc highgui videoio)

    set(_ocv_found FALSE)
    if(TARGET XIMEA::xiAPI AND XIMEA_OPENCV_WRAPPER_DIR
       AND EXISTS "${XIMEA_OPENCV_WRAPPER_DIR}/xiApiPlusOcv.cpp" AND OpenCV_FOUND)
        set(_ocv_found TRUE)
    endif()
    set(XIMEA_OPENCV_FOUND ${_ocv_found} PARENT_SCOPE)

    if(NOT _ocv_found)
        message(WARNING
            "OpenCV or the XIMEA xiAPIplus OpenCV wrapper not found, skipping "
            "${PROJECT_NAME}. Install OpenCV 4 (core, imgproc, highgui, videoio) "
            "and point OpenCV_DIR at its CMake package, or install it with the "
            "system package manager (libopencv-dev on Debian/Ubuntu). "
            "xiApiPlusOcv.hpp is expected at "
            "<XIMEA_SP_PATH>/Examples/Sources/_libs/xiAPIplusOpenCV on Windows or "
            "/opt/XIMEA/samples/_libs/xiAPIplusOpenCV on Linux.")
        _ximea_sample_skip_marker(REASON "OpenCV or the xiAPIplus OpenCV wrapper not found")
        return()
    endif()
    _ximea_sample_skip_marker()

    if(NOT TARGET XIMEA_xiAPIplusOpenCV)
        add_library(XIMEA_xiAPIplusOpenCV STATIC "${XIMEA_OPENCV_WRAPPER_DIR}/xiApiPlusOcv.cpp")
        target_compile_features(XIMEA_xiAPIplusOpenCV PUBLIC cxx_std_17)
        target_include_directories(XIMEA_xiAPIplusOpenCV PUBLIC "${XIMEA_OPENCV_WRAPPER_DIR}")
        if(NOT WIN32)
            target_compile_definitions(XIMEA_xiAPIplusOpenCV PRIVATE UNIX)
        endif()
        target_link_libraries(XIMEA_xiAPIplusOpenCV PUBLIC XIMEA::xiAPI ${OpenCV_LIBS})
        add_library(XIMEA::xiAPIplusOpenCV ALIAS XIMEA_xiAPIplusOpenCV)
    endif()
endfunction()

# Enables the CUDA language and finds the CUDA toolkit, which provides
# CUDA::cudart. A macro, because enable_language() must run in the sample's own
# directory scope. A sample checks APPLICATION_SAMPLES_CUDA_FOUND and skips
# itself when the toolkit or a working CUDA compiler is missing.
macro(application_samples_find_cuda)
    # Lets CMake pick the compiler's default GPU architecture instead of
    # warning that CUDA_ARCHITECTURES is empty.
    if(POLICY CMP0104)
        cmake_policy(SET CMP0104 NEW)
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    set(APPLICATION_SAMPLES_CUDA_FOUND FALSE)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(APPLICATION_SAMPLES_CUDA_FOUND TRUE)
        endif()
    endif()

    if(APPLICATION_SAMPLES_CUDA_FOUND)
        _ximea_sample_skip_marker()
    else()
        message(WARNING
            "CUDA toolkit not found, skipping ${PROJECT_NAME}. Install the NVIDIA "
            "CUDA Toolkit and make sure nvcc is on PATH or CUDA_PATH is set. With a "
            "Visual Studio generator, the toolkit's Visual Studio integration must "
            "be installed as well; otherwise configure with -G Ninja from a "
            "developer command prompt.")
        _ximea_sample_skip_marker(REASON "CUDA toolkit not found")
    endif()
endmacro()

# Finds an OpenCL SDK (headers and ICD loader) with find_package(OpenCL), which
# defines OpenCL::OpenCL. GPU vendor SDKs, including the CUDA toolkit, ship
# one. A sample checks APPLICATION_SAMPLES_OPENCL_FOUND and skips itself when
# it is missing.
function(application_samples_find_opencl)
    find_package(OpenCL QUIET)
    set(APPLICATION_SAMPLES_OPENCL_FOUND ${OpenCL_FOUND} PARENT_SCOPE)

    if(NOT OpenCL_FOUND)
        message(WARNING
            "OpenCL not found, skipping ${PROJECT_NAME}. Install an OpenCL SDK: "
            "the GPU vendor's SDK (the CUDA toolkit for NVIDIA) on Windows, or the "
            "OpenCL headers and ICD loader on Linux (opencl-headers and "
            "ocl-icd-opencl-dev on Debian/Ubuntu).")
        _ximea_sample_skip_marker(REASON "OpenCL not found")
        return()
    endif()
    _ximea_sample_skip_marker()
endfunction()

# Finds the MultiRecorder sources that ship with the XIMEA Software Package
# under Examples/MultiRecorder/sources (examples/MultiRecorder/sources on Linux)
# and builds the parts a camera server or client needs as XIMEA::MultiRecorder.
# The OpenCV live preview is loaded at run time only if OpenCV is installed, and
# recording to FIS storage and TIFF export are left out. Call ximea_find_sdk()
# first; a sample checks XIMEA_MULTIRECORDER_FOUND and skips itself when the SP
# component is missing.
function(ximea_find_multirecorder)
    set(_ximea_sp_path "$ENV{XIMEA_SP_PATH}")
    set(_mr_roots "${_ximea_sp_path}" "${XIMEA_ROOT}" "$ENV{XIMEA_ROOT}")
    set(_mr_hints)
    foreach(_root IN LISTS _mr_roots)
        if(_root)
            list(APPEND _mr_hints
                "${_root}/Examples/MultiRecorder/sources"
                "${_root}/examples/MultiRecorder/sources")
        endif()
    endforeach()
    if(NOT WIN32)
        list(APPEND _mr_hints "/opt/XIMEA/examples/MultiRecorder/sources")
    endif()

    find_path(XIMEA_MULTIRECORDER_SOURCE_DIR NAMES RecorderApi.h HINTS ${_mr_hints})
    mark_as_advanced(XIMEA_MULTIRECORDER_SOURCE_DIR)
    find_package(Threads QUIET)

    set(_mr_dir "${XIMEA_MULTIRECORDER_SOURCE_DIR}")
    set(_mr_found FALSE)
    if(TARGET XIMEA::xiAPI AND Threads_FOUND AND _mr_dir
       AND EXISTS "${_mr_dir}/tinyxml/tinyxml.cpp"
       AND EXISTS "${_mr_dir}/xiApi/xiApiEnumerators.cpp"
       AND EXISTS "${_mr_dir}/httplib/httplib.h")
        set(_mr_found TRUE)
    endif()
    set(XIMEA_MULTIRECORDER_FOUND ${_mr_found} PARENT_SCOPE)

    if(NOT _mr_found)
        message(WARNING
            "XIMEA MultiRecorder sources not found, skipping ${PROJECT_NAME}. "
            "Install the XIMEA Software Package with its examples; RecorderApi.h "
            "is expected at <XIMEA_SP_PATH>/Examples/MultiRecorder/sources on "
            "Windows or /opt/XIMEA/examples/MultiRecorder/sources on Linux.")
        _ximea_sample_skip_marker(REASON "XIMEA MultiRecorder sources not found")
        return()
    endif()
    _ximea_sample_skip_marker()

    if(NOT TARGET XIMEA_MultiRecorder)
        # Same server and client sources as the SP's camera-side processing
        # example, without the MultiRecorder application itself.
        add_library(XIMEA_MultiRecorder STATIC EXCLUDE_FROM_ALL
            "${_mr_dir}/CameraClient.cpp"
            "${_mr_dir}/CameraControl.cpp"
            "${_mr_dir}/CameraServer.cpp"
            "${_mr_dir}/ImageExporter.cpp"
            "${_mr_dir}/LiveImagePreview.cpp"
            "${_mr_dir}/RecorderApi.cpp"
            "${_mr_dir}/RemoteCameraClient.cpp"
            "${_mr_dir}/UserImageProcessor.cpp"
            "${_mr_dir}/tinyxml/tinyxml.cpp"
            "${_mr_dir}/tinyxml/tinyxmlerror.cpp"
            "${_mr_dir}/tinyxml/tinyxmlparser.cpp"
            "${_mr_dir}/xiApi/xiApiEnumerators.cpp"
        )
        target_compile_features(XIMEA_MultiRecorder PUBLIC cxx_std_17)
        # The SDK's xiApi.h comes first; the sources folder adds the
        # xiApiParams.h and xiApiEnumerators.h helpers it ships alongside.
        target_include_directories(XIMEA_MultiRecorder
            PUBLIC "${XIMEA_INCLUDE_DIR}" "${_mr_dir}" "${_mr_dir}/tinyxml"
            PRIVATE "${_mr_dir}/xiApi")
        # TIXML_USE_STL changes the TinyXML class layout, so consumers that
        # include tinyxml.h need it too.
        target_compile_definitions(XIMEA_MultiRecorder
            PUBLIC TIXML_USE_STL
            PRIVATE
                CPPHTTPLIB_CONNECTION_TIMEOUT_SECOND=5
                CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND=30
                CPPHTTPLIB_WRITE_TIMEOUT_SECOND=30
                CPPHTTPLIB_READ_TIMEOUT_SECOND=30)
        if(MSVC)
            target_compile_definitions(XIMEA_MultiRecorder PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        if(APPLE)
            target_compile_definitions(XIMEA_MultiRecorder PUBLIC MACOS)
        elseif(UNIX)
            target_compile_definitions(XIMEA_MultiRecorder PUBLIC LINUX)
        endif()
        target_link_libraries(XIMEA_MultiRecorder PUBLIC XIMEA::xiAPI Threads::Threads)
        if(WIN32)
            target_link_libraries(XIMEA_MultiRecorder PUBLIC ws2_32)
        else()
            target_link_libraries(XIMEA_MultiRecorder PUBLIC ${CMAKE_DL_LIBS})
        endif()
        add_library(XIMEA::MultiRecorder ALIAS XIMEA_MultiRecorder)
    endif()
endfunction()
