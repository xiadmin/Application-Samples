cmake_minimum_required(VERSION 3.16)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
    project(ximea-sample-cmake-helpers LANGUAGES C CXX)
endif()

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
            find_package(TIFF QUIET)

            set(_ximea_plus_sources "${XIMEA_INCLUDE_PLUS_DIR}/xiAPIplus_core.cpp")
            if(TIFF_FOUND)
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

                if(TIFF_FOUND)
                    target_link_libraries(XIMEA_xiAPIplus PUBLIC TIFF::TIFF)
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
