"""
Sample name: Display-image-opencv
Category: Display / OpenCV integration
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures one MONO8, RGB24 or MONO16 image and shows it in
an OpenCV window scaled to fit the screen.
"""

import argparse
import sys

from ximea import xiapi

IMAGE_FORMATS = {
    "mono8": "XI_MONO8",
    "rgb24": "XI_RGB24",
    "mono16": "XI_MONO16",
}
DEFAULT_EXPOSURE_US = 10_000
GRAB_TIMEOUT_MS = 5_000
WHITE_BALANCE_FRAME_COUNT = 10
MAX_WINDOW_WIDTH = 1920
MAX_WINDOW_HEIGHT = 1280
WINDOW_NAME = "XIMEA OpenCV image"
KEY_POLL_MS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one image and show it in an OpenCV window.")
    parser.add_argument(
        "--format",
        choices=IMAGE_FORMATS,
        default="mono8",
        help="image data format (default: mono8); rgb24 needs a color camera",
    )
    parser.add_argument(
        "--exposure-us",
        type=int,
        default=DEFAULT_EXPOSURE_US,
        help=f"exposure time in microseconds (default: {DEFAULT_EXPOSURE_US})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    is_rgb = args.format == "rgb24"

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
            import cv2
        except ImportError:
            print(
                "Error: opencv-python is not installed. Run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 1

        print("Opening camera index 0")
        cam.open_device()
        is_device_open = True

        if is_rgb and not cam.is_iscolor():
            print("Error: --format rgb24 needs a color camera", file=sys.stderr)
            return 1

        cam.set_imgdataformat(IMAGE_FORMATS[args.format])
        cam.set_exposure(args.exposure_us)
        if is_rgb:
            cam.enable_auto_wb()

        cam.start_acquisition()
        is_acquiring = True

        img = xiapi.Image()
        if is_rgb:
            # Discard the first frames while auto white balance settles.
            for _ in range(WHITE_BALANCE_FRAME_COUNT):
                cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
        cam.get_image(img, timeout=GRAB_TIMEOUT_MS)

        # The array is a copy, so it stays valid after the camera is closed.
        # xiAPI delivers XI_RGB24 as BGR, which is the order OpenCV expects.
        data = img.get_image_data_numpy()
        print(f"Captured {args.format} image {img.width}x{img.height}")
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

    if args.format == "mono16":
        # MONO16 holds 10/12-bit sensor values, which cv2.imshow shows almost
        # black. Stretch the captured range to 8 bits for display.
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    height, width = data.shape[:2]
    scale = min(1.0, MAX_WINDOW_WIDTH / width, MAX_WINDOW_HEIGHT / height)

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, int(width * scale), int(height * scale))
        cv2.imshow(WINDOW_NAME, data)
        print("Press any key in the window or close it to exit")
        # waitKey(0) would keep waiting after the window is closed with its X button.
        while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(KEY_POLL_MS) != -1:
                break
        cv2.destroyAllWindows()
    except cv2.error as error:
        # opencv-python-headless and systems without a display have no windows.
        print(f"Error: could not show the OpenCV window: {error}", file=sys.stderr)
        return 1

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
