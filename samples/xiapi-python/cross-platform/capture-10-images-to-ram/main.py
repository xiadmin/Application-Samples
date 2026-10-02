"""
Sample name: Capture-10-images-to-ram
Category: Basic acquisition / image capture
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures 10 images into application-owned RAM.
"""

import sys
from dataclasses import dataclass

from ximea import xiapi

FRAME_COUNT = 10
EXPOSURE_US = 10_000
GRAB_TIMEOUT_MS = 5_000


@dataclass(frozen=True)
class StoredFrame:
    """A frame copied out of the driver buffer, owned by the application."""

    width: int
    height: int
    frame_number: int
    pixels: bytes


def main() -> int:
    cam = xiapi.Camera()
    is_device_open = False
    is_acquiring = False
    frames: list[StoredFrame] = []
    ret = 0

    try:
        if cam.get_number_devices() == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1

        print("Opening camera index 0")
        cam.open_device()
        is_device_open = True

        cam.set_imgdataformat("XI_RAW8")
        cam.set_exposure(EXPOSURE_US)

        cam.start_acquisition()
        is_acquiring = True

        img = xiapi.Image()
        for _ in range(FRAME_COUNT):
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
            # img.bp points into a driver buffer that the next get_image() call
            # reuses and close_device() frees; get_image_data_raw() copies it.
            frames.append(StoredFrame(img.width, img.height, img.nframe, img.get_image_data_raw()))
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        ret = 1
    finally:
        if is_acquiring:
            cam.stop_acquisition()
        if is_device_open:
            cam.close_device()

    if ret != 0:
        return ret

    print("Camera closed; reading retained images from RAM")
    for i, frame in enumerate(frames):
        first_byte = frame.pixels[0] if frame.pixels else -1
        print(
            f"Frame {i + 1}/{FRAME_COUNT}: {frame.width}x{frame.height} "
            f"nframe={frame.frame_number} first_byte={first_byte}"
        )

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
