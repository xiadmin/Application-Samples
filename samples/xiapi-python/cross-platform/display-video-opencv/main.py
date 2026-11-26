"""
Sample name: Display-video-opencv
Category: Display / OpenCV integration
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Shows a live MONO8 or RGB24 video in an OpenCV window with
the elapsed time drawn on each frame until ESC is pressed.
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

from ximea import xiapi

IMAGE_FORMATS = {
    "mono8": "XI_MONO8",
    "rgb24": "XI_RGB24",
}
DEFAULT_EXPOSURE_US = 20_000
GRAB_TIMEOUT_MS = 5_000
MAX_WINDOW_WIDTH = 1920
MAX_WINDOW_HEIGHT = 1280
WINDOW_NAME = "XIMEA OpenCV video"
ESC_KEY = 27
TEXT_COLOR = (255, 255, 255)
# Text height as a fraction of the frame height, so the overlay looks the
# same at every resolution.
TEXT_HEIGHT_RATIO = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show live video in an OpenCV window.")
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


def install_requirements(package: str) -> bool:
    """Install this sample's requirements.txt if the user allowed downloads.

    Runs only when the environment variable AUTOMATICALLY_DOWNLOAD_DEPENDENCIES
    is 1, ON, TRUE or YES; pip then accepts only packages whose hashes match
    dependencies/python-constraints.txt. Returns whether the install ran and
    succeeded; otherwise prints how to provide the package to stderr. A copy
    the user installs themselves is not checked.
    """
    requirements = Path(__file__).resolve().parent / "requirements.txt"
    command = [sys.executable, "-m", "pip", "install", "--require-hashes", "-r", str(requirements)]
    command_line = subprocess.list2cmdline(command)
    setting = os.environ.get("AUTOMATICALLY_DOWNLOAD_DEPENDENCIES", "").strip().upper()
    should_download = setting in ("1", "ON", "TRUE", "YES")

    if not should_download:
        python = subprocess.list2cmdline([sys.executable])
        print(
            f"Error: {package} is not installed, and nothing is downloaded unless you ask for it.\n"
            "Either install it yourself, any way you like (your copy is not checked), e.g.\n"
            f"  {python} -m pip install {package}\n"
            "or the tested versions, hash-checked (pip 26.3 or newer):\n"
            f"  {command_line}\n"
            "or set the environment variable AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1 and run\n"
            "the sample again, so that it runs the hash-checked command itself:\n"
            "  bash:       export AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1\n"
            "  PowerShell: $env:AUTOMATICALLY_DOWNLOAD_DEPENDENCIES = '1'\n"
            'See "Third-party libraries" in README.md.',
            file=sys.stderr,
        )
        return False

    print(
        f"{package} is not installed and AUTOMATICALLY_DOWNLOAD_DEPENDENCIES is set;\n"
        f"installing the hash-checked requirements:\n  {command_line}",
        file=sys.stderr,
    )
    result = subprocess.run(command, cwd=requirements.parent, check=False)
    if result.returncode != 0:
        print(
            f"Error: pip failed with exit code {result.returncode}; see its message above.\n"
            "It needs pip 26.3 or newer and a virtual environment it may install into.\n"
            'See "Third-party libraries" in README.md.',
            file=sys.stderr,
        )
        return False

    importlib.invalidate_caches()
    return True


def main() -> int:
    args = parse_args()
    is_rgb = args.format == "rgb24"

    cam = xiapi.Camera()
    is_device_open = False
    is_acquiring = False
    has_window = False
    ret = 0
    frame_count = 0
    elapsed_s = 0.0

    try:
        if cam.get_number_devices() == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1

        # Imported after the camera check, so a machine without a camera
        # reports that first even when the requirements are not installed.
        try:
            import cv2
        except ImportError:
            if not install_requirements("opencv-python"):
                return 1
            import cv2

        try:
            import numpy
        except ImportError:
            if not install_requirements("numpy"):
                return 1
            import numpy
        # ximea.xiapi looks for numpy only when it is first imported; hand it the
        # copy that may have been installed just now.
        xiapi.np = numpy

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

        # cv2 is bound from here on, so its errors can be caught.
        try:
            img = xiapi.Image()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            has_window = True
            print("Showing video. Press ESC in the window or close it to exit")
            start_time = time.monotonic()
            while True:
                cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
                # xiAPI delivers XI_RGB24 as BGR, which is the order OpenCV expects.
                data = img.get_image_data_numpy()
                height, width = data.shape[:2]

                if frame_count == 0:
                    scale = min(1.0, MAX_WINDOW_WIDTH / width, MAX_WINDOW_HEIGHT / height)
                    cv2.resizeWindow(WINDOW_NAME, int(width * scale), int(height * scale))

                elapsed_s = time.monotonic() - start_time
                # FONT_HERSHEY_SIMPLEX is about 22 px high at scale 1.
                font_scale = height * TEXT_HEIGHT_RATIO / 22
                thickness = max(1, round(font_scale * 2))
                cv2.putText(
                    data,
                    f"{elapsed_s:6.2f} s",
                    (width // 20, height // 20 + round(font_scale * 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    TEXT_COLOR,
                    thickness,
                )
                cv2.imshow(WINDOW_NAME, data)
                frame_count += 1

                if cv2.waitKey(1) == ESC_KEY:
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except cv2.error as error:
            # opencv-python-headless and systems without a display have no windows.
            print(f"Error: could not show the OpenCV window: {error}", file=sys.stderr)
            ret = 1
    except KeyboardInterrupt:
        print("Interrupted")
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        ret = 1
    finally:
        if is_acquiring:
            cam.stop_acquisition()
        if is_device_open:
            cam.close_device()

    if has_window:
        cv2.destroyAllWindows()

    if ret == 0:
        if elapsed_s > 0:
            print(f"Showed {frame_count} frames in {elapsed_s:.1f} s ({frame_count / elapsed_s:.1f} fps)")
        print("Done")

    return ret


if __name__ == "__main__":
    sys.exit(main())
