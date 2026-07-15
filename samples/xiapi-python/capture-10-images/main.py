"""main - XIMEA xiAPI capture sample (Python 3.11+)

Opens the first available XIMEA camera, sets exposure to 100 ms,
captures 10 frames, prints per-frame metadata, then closes.

Build: no build step needed — run directly with Python.
"""

import sys
from ximea import xiapi

frame_count = 10
exposure_us = 100000  
grab_timeout_ms = 5000


def main():
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

        cam.set_exposure(exposure_us)
        print(f"Exposure: {exposure_us} us ({exposure_us // 1000} ms)")

        cam.start_acquisition()
        is_acquiring = True

        print(f"Capturing {frame_count} frames")

        img = xiapi.Image()
        for i in range(frame_count):
            current_frame = i + 1
            cam.get_image(img, timeout=grab_timeout_ms)
            data = img.get_image_data_numpy()
            first_byte = int(data.flat[0]) if data is not None and data.size > 0 else -1
            print(
                f"Frame {current_frame}/{frame_count}: "
                f"{img.width}x{img.height} "
                f"nframe={img.nframe} "
                f"first_byte={first_byte}"
            )
    except xiapi.Xi_error as e:
        if current_frame is not None:
            print(f"Error: failed on frame {current_frame}/{frame_count}: {e}", file=sys.stderr)
        else:
            print(f"Error: xiAPI call failed: {e}", file=sys.stderr)
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
