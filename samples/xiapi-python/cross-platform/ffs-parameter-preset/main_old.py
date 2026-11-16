"""
Sample name: ffs-parameter-preset
Category: Firmware / file access / LUT
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Saves a name=value parameter preset into the camera Flash
File System (FFS) file User1, shows it, and applies it back to the camera.
"""

# Usage:
#   python main.py [-s SERIAL] save name=value [name=value ...]
#   python main.py [-s SERIAL] show     (also the default without a command)
#   python main.py [-s SERIAL] load
#
# Example (per-camera digital gain):
#   python main.py save gain_selector=XI_GAIN_SELECTOR_DIGITAL_ALL gain=0.91
#   python main.py load
#
# Notes:
# - Names are xiAPI parameter strings ("gain", "exposure", "gain_selector", ...).
#   Enumerated values are written by name, e.g. XI_GAIN_SELECTOR_DIGITAL_ALL.
# - Pairs are applied in the order they were saved, so put a selector before
#   the parameter it selects (gain_selector before gain).
# - Some parameters reset others (imgdataformat or downsampling can change gain
#   or ROI). Put them first in the preset, or load the preset after your
#   application has set them.
# - "save" replaces the whole content of the FFS file.

import argparse
import contextlib
import sys
from collections.abc import Iterator

from ximea import xiapi, xidefs

# MX, MC, CB and MT cameras accept application data only in the FFS files
# User1, User2 and User3. Using User1 keeps the sample working on those
# families as well as on newer camera models.
FFS_FILE_NAME = "User1"
PRESET_HEADER = "# XIMEA parameter preset"
MAX_PRESET_BYTES = 4096

# XI_RETURN code from xiApi.h; the Python package lists it only as a
# description in ximea.xidefs.ERROR_CODES.
XI_FFS_FILE_NOT_FOUND = 113

ParamValue = bool | int | float | str
Pair = tuple[str, str]


class PresetError(ValueError):
    """A preset entry or the stored preset text is not usable."""


def parse_pair(item: str) -> Pair:
    name, separator, text = item.partition("=")
    name = name.strip()
    if not separator or not name:
        raise PresetError(f"expected name=value, got '{item}'")
    return name, text.strip()


def parse_value(name: str, text: str) -> ParamValue:
    """Converts preset text to the Python type that set_param expects for name."""
    # A parameter may carry a modifier such as "exposure:direct_update".
    base_name = name.split(":", 1)[0]
    value_type = xidefs.VAL_TYPE.get(base_name)
    if value_type is None:
        raise PresetError(f"unknown xiAPI parameter '{name}'")

    if value_type == "xiTypeEnum":
        allowed = xidefs.ASSOC_ENUM[base_name]
        if text not in allowed:
            raise PresetError(f"'{text}' is not a value of {name}; use one of: {', '.join(allowed)}")
        return text
    if value_type == "xiTypeString":
        return text

    try:
        if value_type == "xiTypeFloat":
            return float(text)
        if value_type == "xiTypeInteger":
            return int(text)
    except ValueError as error:
        raise PresetError(f"{name} needs a number, got '{text}'") from error

    if value_type == "xiTypeBoolean":
        lowered = text.lower()
        if lowered in ("1", "true"):
            return True
        if lowered in ("0", "false"):
            return False
        raise PresetError(f"{name} needs 0, 1, true or false, got '{text}'")

    # xiTypeInteger64 (read-only counters) and xiTypeCommand (actions) do not
    # belong in a preset; the Python set_param also has no Integer64 branch.
    raise PresetError(f"{name} is of type {value_type}, which a preset cannot store")


def format_preset(pairs: list[Pair]) -> str:
    lines = [PRESET_HEADER, *(f"{name}={text}" for name, text in pairs)]
    preset_text = "\n".join(lines) + "\n"

    # Keep the text ASCII: the Python set_param passes len(str) as the length,
    # which equals the byte count only for ASCII text.
    if not preset_text.isascii():
        raise PresetError("the preset must contain ASCII characters only")
    if len(preset_text) > MAX_PRESET_BYTES:
        raise PresetError(f"the preset is {len(preset_text)} bytes; the limit is {MAX_PRESET_BYTES}")
    return preset_text


def parse_preset(preset_text: str) -> list[Pair]:
    pairs = []
    for line_number, line in enumerate(preset_text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            pairs.append(parse_pair(line))
        except PresetError as error:
            raise PresetError(f"stored preset line {line_number}: {error}") from error
    return pairs


def read_ffs_text(cam: xiapi.Camera) -> str | None:
    """Returns the content of the preset file, or None when it does not exist."""
    cam.set_ffs_file_name(FFS_FILE_NAME)
    try:
        data = cam.get_read_file_ffs(buffer_size=MAX_PRESET_BYTES + 1)
    except xiapi.Xi_error as error:
        if error.status == XI_FFS_FILE_NOT_FOUND:
            return None
        raise
    # get_param returns the raw string buffer, terminated by NUL.
    return data.split(b"\0", 1)[0].decode("ascii", errors="replace")


def write_ffs_text(cam: xiapi.Camera, preset_text: str) -> None:
    cam.set_ffs_file_name(FFS_FILE_NAME)
    cam.set_write_file_ffs(preset_text)


def apply_pairs(cam: xiapi.Camera, pairs: list[Pair]) -> None:
    for name, text in pairs:
        try:
            cam.set_param(name, parse_value(name, text))
        except xiapi.Xi_error as error:
            raise PresetError(f"the camera rejected {name}={text}: {error}") from error
        print(f"  {name} = {text}")


def save_preset(cam: xiapi.Camera, pairs: list[Pair]) -> None:
    preset_text = format_preset(pairs)

    # Applying first proves that this camera accepts every pair before it is stored.
    print("Applying to the camera:")
    apply_pairs(cam, pairs)

    write_ffs_text(cam, preset_text)
    if read_ffs_text(cam) != preset_text:
        raise PresetError(f"verification failed: '{FFS_FILE_NAME}' does not contain the saved preset")
    print(f"Preset saved to FFS file '{FFS_FILE_NAME}'")


def show_preset(cam: xiapi.Camera) -> None:
    preset_text = read_ffs_text(cam)
    if preset_text is None:
        print(f"No preset stored in FFS file '{FFS_FILE_NAME}'")
        return
    print(f"Preset in FFS file '{FFS_FILE_NAME}':")
    for name, text in parse_preset(preset_text):
        print(f"  {name} = {text}")


def load_preset(cam: xiapi.Camera) -> None:
    preset_text = read_ffs_text(cam)
    if preset_text is None:
        raise PresetError(f"no preset stored in FFS file '{FFS_FILE_NAME}'; run 'save' first")
    print(f"Applying preset from FFS file '{FFS_FILE_NAME}':")
    apply_pairs(cam, parse_preset(preset_text))
    print("Preset applied")


@contextlib.contextmanager
def open_camera(serial: str | None) -> Iterator[xiapi.Camera]:
    cam = xiapi.Camera()
    if serial:
        print(f"Opening camera {serial}")
        cam.open_device_by_SN(serial)
    else:
        print("Opening camera index 0")
        cam.open_device()
    try:
        yield cam
    finally:
        cam.close_device()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save, show or load a parameter preset in the camera FFS.")
    parser.add_argument("-s", "--serial", help="serial number of the camera; default is the first camera")
    commands = parser.add_subparsers(dest="command")
    save_parser = commands.add_parser("save", help="apply name=value pairs and store them in the camera")
    save_parser.add_argument("pairs", nargs="+", metavar="name=value")
    commands.add_parser("show", help="print the stored preset (default)")
    commands.add_parser("load", help="apply the stored preset to the camera")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "show"

    try:
        pairs: list[Pair] = []
        if command == "save":
            # Validate everything before touching the camera.
            pairs = [parse_pair(item) for item in args.pairs]
            for name, text in pairs:
                parse_value(name, text)
            format_preset(pairs)

        if xiapi.Camera().get_number_devices() == 0:
            print("Error: no XIMEA cameras detected", file=sys.stderr)
            return 1

        with open_camera(args.serial) as cam:
            if command == "save":
                save_preset(cam, pairs)
            elif command == "show":
                show_preset(cam)
            else:
                load_preset(cam)
    except PresetError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
