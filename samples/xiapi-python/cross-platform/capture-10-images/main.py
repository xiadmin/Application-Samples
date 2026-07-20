"""
Sample name: Capture-10-images
Category: Basic acquisition / image capture
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Captures 10 images.

Copyright (c) 2026 XIMEA s.r.o.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import sys

from ximea import xiapi

FRAME_COUNT = 10
EXPOSURE_US = 100_000
GRAB_TIMEOUT_MS = 5_000


def main() -> int:
    cam = xiapi.Camera()
    is_device_open = False
    is_acquiring = False
    current_frame = None
    ret = 0

    try:
        count = cam.get_number_devices()
        if count == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1

        print(f"Found {count} camera(s), opening index 0")

        cam.open_device()
        is_device_open = True

        cam.set_exposure(EXPOSURE_US)
        print(f"Exposure: {EXPOSURE_US} us ({EXPOSURE_US // 1000} ms)")

        cam.start_acquisition()
        is_acquiring = True

        print(f"Capturing {FRAME_COUNT} frames")

        img = xiapi.Image()
        for i in range(FRAME_COUNT):
            current_frame = i + 1
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
            data = img.get_image_data_numpy()
            first_byte = int(data.flat[0]) if data is not None and data.size > 0 else -1
            print(
                f"Frame {current_frame}/{FRAME_COUNT}: "
                f"{img.width}x{img.height} "
                f"nframe={img.nframe} "
                f"first_byte={first_byte}"
            )
    except xiapi.Xi_error as error:
        if current_frame is not None:
            print(f"Error: failed on frame {current_frame}/{FRAME_COUNT}: {error}", file=sys.stderr)
        else:
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
