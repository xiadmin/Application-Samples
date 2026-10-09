"""
Sample name: Capture-from-two-cameras
Category: Multi-camera / basic acquisition
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Opens the first two connected cameras, splits the interface
bandwidth between them, captures one image from each and prints its size and
first pixels.
"""

import argparse
import sys

from ximea import xiapi

CAMERA_COUNT = 2
DEFAULT_EXPOSURE_US = 10_000
GRAB_TIMEOUT_MS = 5_000
PREVIEW_PIXEL_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one image from each of two cameras.")
    parser.add_argument(
        "--exposure-us",
        type=int,
        default=DEFAULT_EXPOSURE_US,
        help=f"exposure time of both cameras in microseconds (default: {DEFAULT_EXPOSURE_US})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cameras = [xiapi.Camera(dev_id=i) for i in range(CAMERA_COUNT)]
    open_cameras: list[xiapi.Camera] = []
    acquiring_cameras: list[xiapi.Camera] = []
    ret = 0

    try:
        device_count = cameras[0].get_number_devices()
        if device_count == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1
        if device_count < CAMERA_COUNT:
            print(
                f"Error: {CAMERA_COUNT} cameras are required, {device_count} detected",
                file=sys.stderr,
            )
            return 1

        print("Opening cameras...")
        for cam in cameras:
            cam.open_device()
            open_cameras.append(cam)

        # Both cameras usually share one USB/PCIe controller, so each gets an
        # equal part of the bandwidth the first camera measured.
        camera_bandwidth_mbps = int(cameras[0].get_limit_bandwidth() / CAMERA_COUNT)
        for i, cam in enumerate(cameras, start=1):
            cam.set_limit_bandwidth_mode("XI_ON")
            cam.set_limit_bandwidth(camera_bandwidth_mbps)
            cam.set_imgdataformat("XI_MONO8")
            cam.set_exposure(args.exposure_us)
            serial_number = cam.get_device_sn().decode("utf-8")
            print(
                f"Camera {i}: serial number {serial_number}, "
                f"bandwidth limit {camera_bandwidth_mbps} Mbit/s, "
                f"exposure {cam.get_exposure()} us"
            )

        print("Starting data acquisition...")
        for cam in cameras:
            cam.start_acquisition()
            acquiring_cameras.append(cam)

        for i, cam in enumerate(cameras, start=1):
            img = xiapi.Image()
            cam.get_image(img, timeout=GRAB_TIMEOUT_MS)
            # MONO8 stores one byte per pixel; the slice stays inside the first row.
            first_pixels = list(img.get_image_data_raw()[:PREVIEW_PIXEL_COUNT])
            print(f"Camera {i}: image {img.width}x{img.height} received")
            print(f"Camera {i}: first {PREVIEW_PIXEL_COUNT} pixels: {first_pixels}")
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        ret = 1
    finally:
        # Release every camera, even when stopping or closing another one fails.
        for cam in acquiring_cameras:
            try:
                cam.stop_acquisition()
            except xiapi.Xi_error as error:
                print(f"Error: could not stop acquisition: {error}", file=sys.stderr)
                ret = 1
        for cam in open_cameras:
            try:
                cam.close_device()
            except xiapi.Xi_error as error:
                print(f"Error: could not close camera: {error}", file=sys.stderr)
                ret = 1

    if ret == 0:
        print("Done")

    return ret


if __name__ == "__main__":
    sys.exit(main())
