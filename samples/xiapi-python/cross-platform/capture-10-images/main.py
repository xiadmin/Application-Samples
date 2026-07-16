"""main - XIMEA xiAPI capture sample (Python 3.11+)

Opens the first available XIMEA camera, sets exposure to 100 ms,
captures 10 frames, prints per-frame metadata, then closes.

Build: no build step needed — run directly with Python.
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
