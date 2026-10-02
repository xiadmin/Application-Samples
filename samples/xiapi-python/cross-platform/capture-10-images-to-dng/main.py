"""
Sample name: Capture-10-images-to-dng
Category: Basic acquisition / image capture
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures 10 raw images and saves them as DNG files using
the XIMEA DNG Store library.
"""

import sys

from ximea import xiapi

import xiapi_dng

FRAME_COUNT = 10
EXPOSURE_US = 10_000
GRAB_TIMEOUT_MS = 5_000


def main() -> int:
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

        # DNG stores the sensor data as captured: Bayer data on a color camera.
        cam.set_imgdataformat("XI_RAW8")
        cam.set_exposure(EXPOSURE_US)

        cam.start_acquisition()
        is_acquiring = True

        # Read once; the settings do not change between frames here.
        metadata = xiapi_dng.fill_metadata_from_camera(cam)

        img = xiapi.Image()
        for i in range(FRAME_COUNT):
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
            filename = f"image{i:03d}.dng"
            xiapi_dng.store_image(filename, img, metadata)
            print(f"Saved frame {i + 1}/{FRAME_COUNT} ({img.width}x{img.height}) to {filename}")
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        ret = 1
    except OSError as error:
        print(f"Error: could not load the XIMEA DNG Store library: {error}", file=sys.stderr)
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
