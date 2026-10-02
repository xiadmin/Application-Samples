"""ctypes wrapper for the xiapi_dng_store library from the XIMEA Software Package.

Ported from the SP's samples/_libs/xiAPI-Python/xiapi_dng.py. The library is
loaded on first use, from:

- Windows: <XIMEA_SP_PATH>/DNG Store/binX64/xiapi_dng_store.dll
- Linux:   /opt/XIMEA/lib/libxiapi_dng_store.so
- macOS:   /Library/Frameworks/m3api.framework/Libraries/libxiapi_dng_store.dylib
"""

import ctypes
import functools
import os
import platform
from ctypes import POINTER, Structure, byref, c_char, c_char_p, c_float, c_int, c_uint, c_void_p

from ximea import xiapi

XI_OK = 0


class DngMetadata(Structure):
    """Mirror of XI_DNG_METADATA in xiapi_dng_store.h."""

    _fields_ = [
        ("cfa", c_int),  # XI_COLOR_FILTER_ARRAY
        ("maxPixelValue", c_int),
        ("cameraModelName", c_char * 256),
        ("cameraSerialNumber", c_char * 256),
        ("cameraUserId", c_char * 256),
        ("autoExposureOn", c_int),  # 1: on, 0: off, -1: N/A
        ("autoWhiteBalanceOn", c_int),
        ("whiteBalanceR", c_float),
        ("whiteBalanceG", c_float),
        ("whiteBalanceB", c_float),
        ("lensAperture", c_float),
        ("lensFocalLength", c_float),
        ("acqTimeYear", c_int),
        ("acqTimeMonth", c_int),
        ("acqTimeDay", c_int),
        ("acqTimeHour", c_int),
        ("acqTimeMinute", c_int),
        ("acqTimeSecond", c_int),
        # The library writes these two as well; without them it would write
        # past the end of the Python-owned structure.
        ("additionalPrivateData", c_void_p),
        ("additionalPrivateData_size", c_uint),
    ]


@functools.cache
def _library() -> ctypes.CDLL:
    system = platform.system()
    if system == "Windows":
        sp_path = os.environ.get("XIMEA_SP_PATH")
        if not sp_path:
            raise OSError("XIMEA_SP_PATH is not set; install the XIMEA Software Package")
        os.add_dll_directory(os.path.join(sp_path, "DNG Store", "binX64"))
        library = ctypes.CDLL("xiapi_dng_store.dll")
    elif system == "Darwin":
        library = ctypes.CDLL("/Library/Frameworks/m3api.framework/Libraries/libxiapi_dng_store.dylib")
    else:
        library = ctypes.CDLL("/opt/XIMEA/lib/libxiapi_dng_store.so")

    library.xidngFillMetadataFromCameraParams.argtypes = [c_void_p, POINTER(DngMetadata)]
    library.xidngFillMetadataFromCameraParams.restype = c_int
    library.xidngStore.argtypes = [c_char_p, POINTER(xiapi.Image), POINTER(DngMetadata)]
    library.xidngStore.restype = c_int
    return library


def fill_metadata_from_camera(cam: xiapi.Camera) -> DngMetadata:
    """Read the DNG metadata (model, CFA, white balance, ...) from an open camera."""
    metadata = DngMetadata()
    status = _library().xidngFillMetadataFromCameraParams(cam.handle, byref(metadata))
    if status != XI_OK:
        raise xiapi.Xi_error(status)
    return metadata


def store_image(filename: str, img: xiapi.Image, metadata: DngMetadata) -> None:
    """Write img to filename as DNG. Raises xiapi.Xi_error on failure."""
    status = _library().xidngStore(filename.encode("utf-8"), byref(img), byref(metadata))
    if status != XI_OK:
        raise xiapi.Xi_error(status)
