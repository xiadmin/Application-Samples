# cmake/FindXIMEA.cmake
#
# Finds the XIMEA xiAPI SDK and creates the XIMEA::xiAPI imported target.
#
# Search hints (in priority order):
#   1. -DXIMEA_ROOT=<path> on the cmake command line   (requires CMP0074)
#   2. XIMEA_ROOT environment variable                  (requires CMP0074)
#   3. XIMEA_SP_PATH environment variable (Windows SDK installer default)
#   4. Well-known install locations for Linux and macOS
#
# Cache variables set by this module (can be overridden via cmake -D):
#   XIMEA_INCLUDE_DIR      -- directory containing xiApi.h
#   XIMEA_INCLUDE_PLUS_DIR -- directory containing xiApiPlus.h
#   XIMEA_LIBRARY          -- full path to the xiAPI library
#
# Result variable:
#   XIMEA_FOUND        -- TRUE if XIMEA_INCLUDE_DIR, XIMEA_INCLUDE_PLUS_DIR, and XIMEA_LIBRARY were all found

# Honour <PackageName>_ROOT cmake variables and env vars (CMP0074, available since 3.12)
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Collect search hints from environment variables
set(_ximea_sp_path "$ENV{XIMEA_SP_PATH}")

if(WIN32)
    set(_ximea_inc_hints "${_ximea_sp_path}/API/xiAPI")
    set(_ximea_inc_plus_hints "${_ximea_sp_path}/Examples/Sources/_libs/xiAPIplus")
    set(_ximea_lib_hints "${_ximea_sp_path}/API/xiAPI")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_ximea_lib_names xiapi64)
    else()
        set(_ximea_lib_names xiapi32)
    endif()
elseif(APPLE)
    set(_ximea_inc_hints "/Library/Frameworks/m3api.framework/Headers")
    set(_ximea_inc_plus_hints "/Library/Frameworks/m3api.framework/Headers")
    set(_ximea_lib_hints "/Library/Frameworks/m3api.framework")
    set(_ximea_lib_names m3api)
else()  # Linux
    set(_ximea_inc_hints "/opt/XIMEA/include")
    set(_ximea_inc_plus_hints "/opt/XIMEA/include")
    set(_ximea_lib_hints "/opt/XIMEA/lib" "/opt/XIMEA/lib64")
    set(_ximea_lib_names m3api)
endif()

find_path(XIMEA_INCLUDE_DIR
    NAMES xiApi.h
    HINTS ${_ximea_inc_hints}
)

find_path(XIMEA_INCLUDE_PLUS_DIR
    NAMES xiApiPlus.h
    HINTS ${_ximea_inc_plus_hints}
)

find_library(XIMEA_LIBRARY
    NAMES ${_ximea_lib_names}
    HINTS ${_ximea_lib_hints}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XIMEA
    REQUIRED_VARS XIMEA_LIBRARY XIMEA_INCLUDE_DIR XIMEA_INCLUDE_PLUS_DIR
)

# REASON_FAILURE_MESSAGE requires CMake 3.23+; print a hint manually for 3.16 compat.
if(NOT XIMEA_FOUND)
    message(WARNING
        "XIMEA SDK not found. Install the XIMEA Software Package and set "
        "XIMEA_ROOT (cmake -DXIMEA_ROOT=<path>) or the XIMEA_SP_PATH environment variable. "
        "xiApiPlus.h is expected at <XIMEA_SP_PATH>/Examples/Sources/_libs/xiAPIplus "
        "(Windows), /opt/XIMEA/include (Linux), or set XIMEA_INCLUDE_PLUS_DIR manually.")
endif()

if(XIMEA_FOUND AND NOT TARGET XIMEA::xiAPI)
    add_library(XIMEA::xiAPI UNKNOWN IMPORTED)
    set_target_properties(XIMEA::xiAPI PROPERTIES
        IMPORTED_LOCATION "${XIMEA_LIBRARY}"
    )
    target_include_directories(XIMEA::xiAPI INTERFACE "${XIMEA_INCLUDE_DIR}")
endif()

# xiAPIplus -- C++ wrapper with its own include directory.
if(XIMEA_FOUND AND NOT TARGET XIMEA::xiAPIplus)
    add_library(XIMEA::xiAPIplus INTERFACE IMPORTED)
    target_include_directories(XIMEA::xiAPIplus INTERFACE "${XIMEA_INCLUDE_PLUS_DIR}")
    target_link_libraries(XIMEA::xiAPIplus INTERFACE XIMEA::xiAPI)
endif()

# Hide internal cache vars from the default cmake-gui / ccmake view
mark_as_advanced(XIMEA_INCLUDE_DIR XIMEA_INCLUDE_PLUS_DIR XIMEA_LIBRARY)
