# cmake/SampleDefaults.cmake
#
# Modern CMake: Avoid changing global CMAKE_RUNTIME_OUTPUT_DIRECTORY directly inside a module.
# Provide a helper function to set target properties locally instead.

# Derive the canonical sample name from the directory layout:
#   samples/<feature>/<concrete>/<lang>/  -->  <feature>-<concrete>-<lang>
# This matches the name that build.ps1 computes for the output folder.
# CMAKE_CURRENT_LIST_DIR is this module file; CMAKE_SOURCE_DIR is the sample root.
# Walk up from the sample root to find the relative path under samples/.
function(_sample_derive_output_name out_var)
    # Locate the samples/ ancestor by looking for it in CMAKE_SOURCE_DIR
    set(_src "${CMAKE_SOURCE_DIR}")
    string(REPLACE "\\" "/" _src "${_src}")  # normalise Windows separators

    string(REGEX MATCH ".*/samples/(.+)$" _match "${_src}")
    if(_match)
        set(_rel "${CMAKE_MATCH_1}")          # e.g. acquisition/capture-10-images/c
        string(REPLACE "/" "-" _name "${_rel}")  # acquisition-capture-10-images-c
    else()
        # Fallback: just use the immediate directory name
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

    # Set the binary output name to match the build.ps1 folder name convention.
    set_target_properties(${ARG_TARGET} PROPERTIES
        OUTPUT_NAME "${_output_name}"
    )

    # Use CMAKE_BINARY_DIR so the binary lands inside the cmake work directory
    # (e.g. .cmake-tmp/<name>/build/). build.ps1 copies from there to
    # <repo-root>/build/<name>/<output-name>.
    set_target_properties(${ARG_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}/build"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/build"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/build"
    )
endfunction()
