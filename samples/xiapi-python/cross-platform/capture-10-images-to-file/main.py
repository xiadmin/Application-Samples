"""
Sample name: Capture-10-images-to-file
Category: Basic acquisition / image capture
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures 10 images and saves them as TIFF files using
tifffile.
"""

import sys

from ximea import xiapi

FRAME_COUNT = 10
AUTO_EXPOSURE_FRAME_COUNT = 10
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

        # Imported after the camera check, so a machine without a camera
        # reports that first even when the requirements are not installed.
        try:
            import tifffile
        except ImportError:
            print(
                "Error: tifffile is not installed. Run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 1

        print("Opening camera index 0")
        cam.open_device()
        is_device_open = True

        cam.set_exposure(EXPOSURE_US)
        cam.enable_aeag()

        is_color = cam.is_iscolor()
        if is_color:
            cam.set_imgdataformat("XI_RGB24")
            cam.enable_auto_wb()
            cam.set_exp_priority(1.0)
        else:
            cam.set_imgdataformat("XI_MONO8")

        cam.start_acquisition()
        is_acquiring = True

        img = xiapi.Image()

        # Discard the first frames while auto exposure and white balance settle.
        for _ in range(AUTO_EXPOSURE_FRAME_COUNT):
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)

        for i in range(FRAME_COUNT):
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
            # xiAPI delivers XI_RGB24 as BGR; invert_rgb_order gives TIFF its RGB.
            data = img.get_image_data_numpy(invert_rgb_order=is_color)
            filename = f"image{i:03d}.tif"
            tifffile.imwrite(filename, data, photometric="rgb" if is_color else "minisblack")
            print(f"Saved frame {i + 1}/{FRAME_COUNT} to {filename}")
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        ret = 1
    except OSError as error:
        print(f"Error: could not write TIFF file: {error}", file=sys.stderr)
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
