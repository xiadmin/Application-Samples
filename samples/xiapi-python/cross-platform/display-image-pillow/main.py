"""
Sample name: Display-image-pillow
Category: Display / Pillow integration
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures one MONO8, RGB24 or RGB32 image, converts it to a
Pillow image and opens it in the system image viewer.
"""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from ximea import xiapi

# Pillow has no 16-bit RGB modes, and its 16-bit grayscale support is
# limited, so the sample offers only 8-bit-per-channel formats.
IMAGE_FORMATS = {
    "mono8": "XI_MONO8",
    "rgb24": "XI_RGB24",
    "rgb32": "XI_RGB32",
}
EXPOSURE_US = 10_000
GRAB_TIMEOUT_MS = 5_000
WHITE_BALANCE_FRAME_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one image and open it with Pillow.")
    parser.add_argument(
        "--format",
        choices=IMAGE_FORMATS,
        default="mono8",
        help="image data format (default: mono8); rgb24 and rgb32 need a color camera",
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
    is_rgb = args.format != "mono8"

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
            if not install_requirements("pillow"):
                return 1
            from PIL import Image

        print("Opening camera index 0")
        cam.open_device()
        is_device_open = True

        if is_rgb and not cam.is_iscolor():
            print(f"Error: --format {args.format} needs a color camera", file=sys.stderr)
            return 1

        cam.set_imgdataformat(IMAGE_FORMATS[args.format])
        cam.set_exposure(EXPOSURE_US)
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

        # xiAPI delivers RGB formats as BGR(A); Pillow expects RGB(A). The
        # array is a copy, so it stays valid after the camera is closed.
        data = img.get_image_data_numpy(invert_rgb_order=is_rgb)
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

    if args.format == "rgb32":
        # The fourth XI_RGB32 byte is not an opacity value; as alpha it could
        # make the image transparent, so only the color channels are shown.
        data = data[:, :, :3]

    # fromarray picks the mode from the array shape: "L" for MONO8, "RGB" for
    # the color formats.
    picture = Image.fromarray(data)
    print(f"Opening the {picture.mode} image in the system image viewer")
    picture.show()

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
