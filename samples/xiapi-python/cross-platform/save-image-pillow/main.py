"""
Sample name: Save-image-pillow
Category: Image saving / Pillow integration
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures one image (RGB24 on color cameras, MONO8 on
monochrome cameras) and saves it with Pillow, which picks the file format from
the extension.
"""

import argparse
import sys

from ximea import xiapi

DEFAULT_OUTPUT = "image.bmp"
EXPOSURE_US = 10_000
GRAB_TIMEOUT_MS = 5_000
WHITE_BALANCE_FRAME_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one image and save it with Pillow.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output file; the extension selects the format, e.g. .bmp, .png, .jpg (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cam = xiapi.Camera()
    is_device_open = False
    is_acquiring = False
    ret = 0
    data = None

    try:
        if cam.get_number_devices() == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1

        # Imported after the camera check, so a machine without a camera
        # reports that first even when the requirements are not installed.
        try:
            from PIL import Image
        except ImportError:
            print(
                "Error: pillow is not installed. Run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 1

        print("Opening camera index 0")
        cam.open_device()
        is_device_open = True

        cam.set_exposure(EXPOSURE_US)
        is_color = cam.is_iscolor()
        if is_color:
            cam.set_imgdataformat("XI_RGB24")
            cam.enable_auto_wb()
        else:
            cam.set_imgdataformat("XI_MONO8")

        cam.start_acquisition()
        is_acquiring = True

        img = xiapi.Image()
        if is_color:
            # Discard the first frames while auto white balance settles.
            for _ in range(WHITE_BALANCE_FRAME_COUNT):
                cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
        cam.get_image(img, timeout=GRAB_TIMEOUT_MS)

        # xiAPI delivers XI_RGB24 as BGR; invert_rgb_order gives Pillow its RGB.
        # The array is a copy, so it stays valid after the camera is closed.
        data = img.get_image_data_numpy(invert_rgb_order=is_color)
        print(f"Captured image {img.width}x{img.height}")
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

    try:
        Image.fromarray(data).save(args.output)
    except ValueError as error:
        # Raised for an extension Pillow does not know.
        print(f"Error: could not choose a format for {args.output}: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: could not write {args.output}: {error}", file=sys.stderr)
        return 1

    print(f"Saved image to {args.output}")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
