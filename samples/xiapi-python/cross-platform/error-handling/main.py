"""
Sample name: Error-handling
Category: Error handling
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Shows how to detect and recover from an xiAPI error such as
XI_TIMEOUT.
"""

import sys

from ximea import xiapi

# XI_RETURN codes from xiApi.h; the Python package lists them only as
# descriptions in ximea.xidefs.ERROR_CODES.
XI_INVALID_HANDLE = 1
XI_TIMEOUT = 10

EXPOSURE_US = 100_000
# Far shorter than the exposure, so the first frame cannot arrive in time.
SHORT_TIMEOUT_MS = 1
GRAB_TIMEOUT_MS = 5_000


def show_invalid_handle() -> None:
    """A call on a camera that was never opened fails with XI_INVALID_HANDLE."""
    unopened = xiapi.Camera()
    try:
        unopened.get_exposure()
    except xiapi.Xi_error as error:
        if error.status == XI_INVALID_HANDLE:
            print(f"Caught XI_INVALID_HANDLE ({error.status}) from a call without an open camera")
        else:
            print(f"A call without an open camera failed with {error}")
    else:
        print("A call without an open camera succeeded unexpectedly")


def main() -> int:
    # 1. Every failed xiAPI call raises xiapi.Xi_error with the XI_RETURN code
    #    in error.status. This part needs no camera.
    show_invalid_handle()

    cam = xiapi.Camera()
    is_device_open = False
    is_acquiring = False
    ret = 0

    try:
        if cam.get_number_devices() == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1

        print("Opening camera index 0")
        cam.open_device()
        is_device_open = True

        cam.set_exposure(EXPOSURE_US)
        cam.start_acquisition()
        is_acquiring = True

        # 2. Some errors are recoverable. Catch XI_TIMEOUT explicitly, retry
        #    with a timeout that fits the exposure, and re-raise anything else.
        img = xiapi.Image()
        try:
            cam.get_image(img, timeout=SHORT_TIMEOUT_MS)
            print(f"A frame arrived within {SHORT_TIMEOUT_MS} ms; no timeout this time")
        except xiapi.Xi_error as error:
            if error.status != XI_TIMEOUT:
                raise
            print(
                f"Caught XI_TIMEOUT ({error.status}) with a {SHORT_TIMEOUT_MS} ms timeout; "
                f"retrying with {GRAB_TIMEOUT_MS} ms"
            )
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)

        print(f"Got frame {img.width}x{img.height} nframe={img.nframe}")
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        ret = 1
    finally:
        if is_acquiring:
            cam.stop_acquisition()
        if is_device_open:
            cam.close_device()

    if ret == 0:
        print("Done")

    return ret


if __name__ == "__main__":
    sys.exit(main())
