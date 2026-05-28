"""main - XIMEA xiAPI capture sample (Python 3.9+)

Opens the first available XIMEA camera, sets exposure to 100 ms,
captures 10 frames, prints per-frame metadata, then closes.

Build: no build step needed — run directly with Python.
"""

import sys
from ximea import xiapi

frame_count = 10
exposure_us = 100000   # 100 ms in microseconds
grab_timeout_ms = 5000  # must exceed exposure


def run_capture(cam):
    """Set exposure, acquire frames, print per-frame metadata.

    Args:
        cam: Open xiapi.Camera instance.

    Returns:
        0 on success, 1 on failure.
    """
    cam.set_exposure(exposure_us)
    print(f"Exposure: {exposure_us} us ({exposure_us // 1000} ms)")

    cam.start_acquisition()
    print(f"Capturing {frame_count} frames")

    img = xiapi.Image()
    try:
        for i in range(frame_count):
            cam.get_image(img, timeout=grab_timeout_ms)
            data = img.get_image_data_numpy()
            first_byte = int(data.flat[0]) if data is not None and data.size > 0 else -1
            print(
                f"Frame {i + 1}/{frame_count}: "
                f"{img.width}x{img.height} "
                f"nframe={img.nframe} "
                f"first_byte={first_byte}"
            )
    except xiapi.Xi_error as e:
        print(f"Error: failed on frame {i + 1}/{frame_count}: {e}", file=sys.stderr)
        cam.stop_acquisition()
        return 1

    cam.stop_acquisition()
    print("Done")
    return 0


if __name__ == "__main__":
    cam = xiapi.Camera()

    count = cam.get_number_devices()
    if count == 0:
        print("Error: no XIMEA cameras detected", file=sys.stderr)
        sys.exit(1)

    print(f"Found {count} camera(s), opening index 0")

    try:
        cam.open_device()
    except xiapi.Xi_error as e:
        print(f"Error: could not open camera: {e}", file=sys.stderr)
        sys.exit(1)

    ret = run_capture(cam)
    cam.close_device()
    sys.exit(ret)
