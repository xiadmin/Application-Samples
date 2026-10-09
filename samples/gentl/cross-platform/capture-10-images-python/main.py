"""
Sample name: Capture-10-images-python
Category: GenTL / Python consumer
OS platform: Cross-platform
Hardware platform: Cross-platform
API type: GenTL
Short description: Loads the XIMEA GenTL producer with the harvesters GenTL
consumer, sets the exposure and pixel format through the GenICam node map,
captures 10 images and prints the size, timestamp and first pixels of each
one.
"""

import argparse
import os
import sys
from pathlib import Path

FRAME_COUNT = 10
FETCH_TIMEOUT_S = 5.0
DEFAULT_EXPOSURE_US = 10_000.0
FIRST_PIXEL_COUNT = 8
# The current XIMEA producer is preferred over the legacy one.
PRODUCER_FILE_NAMES = ("ximea.gentl2.cti", "ximea.gentl.cti")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture 10 images through the XIMEA GenTL producer with harvesters."
    )
    parser.add_argument(
        "--cti",
        type=Path,
        help="GenTL producer to load. Default: ximea.gentl2.cti or ximea.gentl.cti from "
        "GENICAM_GENTL64_PATH (GENICAM_GENTL32_PATH on 32-bit Python).",
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=DEFAULT_EXPOSURE_US,
        help=f"Exposure time in microseconds. Default: {DEFAULT_EXPOSURE_US:g}.",
    )
    return parser.parse_args()


def find_ximea_producer() -> Path | None:
    """Look for a XIMEA producer in the directories of the standard GenTL variable."""
    variable = "GENICAM_GENTL64_PATH" if sys.maxsize > 2**32 else "GENICAM_GENTL32_PATH"
    search_path = os.environ.get(variable)
    if not search_path:
        return None

    for file_name in PRODUCER_FILE_NAMES:
        for directory in search_path.split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / file_name
            if candidate.is_file():
                return candidate
    return None


def main() -> int:
    args = parse_args()

    cti_path = args.cti or find_ximea_producer()
    if cti_path is None:
        print(
            "Error: no XIMEA GenTL producer found, so no camera device can be opened. "
            "Set GENICAM_GENTL64_PATH or pass --cti <producer.cti>.",
            file=sys.stderr,
        )
        return 1
    if not cti_path.is_file():
        print(
            f"Error: GenTL producer {cti_path} not found, so no camera device can be opened",
            file=sys.stderr,
        )
        return 1

    # Imported after the producer lookup, so a machine without the XIMEA
    # Software Package reports that first even when the requirements are not
    # installed.
    try:
        from genicam.genapi import GenericException as GenApiError
        from genicam.gentl import GenericException as GenTLError
        from harvesters.core import Harvester
    except ImportError:
        print(
            "Error: harvesters is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Loading GenTL producer {cti_path}")
    try:
        with Harvester() as harvester:
            harvester.add_file(str(cti_path), check_existence=True, check_validity=True)
            harvester.update()
            if not harvester.device_info_list:
                print("Error: the GenTL producer found no camera device", file=sys.stderr)
                return 1

            print(f"Opening device {harvester.device_info_list[0].display_name}")
            with harvester.create() as acquirer:
                node_map = acquirer.remote_device.node_map
                node_map.ExposureTime.value = args.exposure_us
                # 8-bit monochrome keeps the printed pixel values simple; the
                # camera's default format is kept when it does not offer it.
                if "Mono8" in node_map.PixelFormat.symbolics:
                    node_map.PixelFormat.value = "Mono8"
                print(f"Pixel format {node_map.PixelFormat.value}, exposure {node_map.ExposureTime.value:g} us")

                acquirer.start()
                for i in range(FRAME_COUNT):
                    # Leaving the with block queues the buffer for the next frame.
                    with acquirer.fetch(timeout=FETCH_TIMEOUT_S) as buffer:
                        component = buffer.payload.components[0]
                        first_pixels = component.data[:FIRST_PIXEL_COUNT].tolist()
                        print(
                            f"Frame {i + 1}/{FRAME_COUNT}: {component.width}x{component.height}, "
                            f"timestamp {buffer.timestamp}, first pixels {first_pixels}"
                        )
                acquirer.stop()
    except (GenTLError, GenApiError) as error:
        print(f"Error: GenTL camera operation failed: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(
            f"Error: could not load GenTL producer {cti_path}, so no camera device can be opened: {error}",
            file=sys.stderr,
        )
        return 1

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
