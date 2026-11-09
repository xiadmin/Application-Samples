"""
Sample name: ffs-parameter-preset
Category: Firmware / file access / LUT
OS platform: Cross-Platform
Hardware platform: Cross-platform
API type: xiAPI Python
Short description: Saves a name=value parameter preset into the camera Flash
File System (FFS) file User1, shows it, and applies it back to the camera.
"""

# Usage: python main.py [-s SERIAL] {save name=value ... | show | load}
#
# - Pairs are applied in the saved order: put a selector before the parameter
#   it selects (gain_selector before gain).
# - User1 is the one FFS file name every camera family accepts (MX, MC, CB and
#   MT allow only User1, User2 and User3).
# - Works on cameras with FFS; tested on MQ (xiQ) and MX (xiX). MQ has no
#   gain_selector (no digital gain), so store gain alone there.

import argparse
import contextlib
import sys
from collections.abc import Iterator

from ximea import xiapi, xidefs

FFS_FILE_NAME = "User1"
PRESET_HEADER = "# XIMEA parameter preset"
MAX_PRESET_BYTES = 4096
# XI_RETURN codes from xiApi.h; the Python package has them only as text.
XI_FFS_FILE_NOT_FOUND = 113
XI_FFS_UNSUPPORTED_CODES = frozenset({12, 100, 106})  # camera has no FFS

Pair = tuple[str, str]


class PresetError(ValueError):
    """A preset entry or the stored preset text is not usable."""


def parse_pair(item: str) -> Pair:
    name, separator, text = item.partition("=")
    if not separator or not name.strip():
        raise PresetError(f"expected name=value, got '{item}'")
    return name.strip(), text.strip()


def parse_value(name: str, text: str) -> bool | int | float | str:
    """Converts preset text to the Python type that set_param expects for name."""
    base_name = name.split(":", 1)[0]  # strip a modifier such as ":direct_update"
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
    if value_type == "xiTypeBoolean":
        if text.lower() not in ("0", "1", "true", "false"):
            raise PresetError(f"{name} needs 0, 1, true or false, got '{text}'")
        return text.lower() in ("1", "true")
    converter = {"xiTypeInteger": int, "xiTypeFloat": float}.get(value_type)
    if converter is None:
        # 64-bit counters (timestamp) and commands do not describe a setting.
        raise PresetError(f"{name} is of type {value_type}, which a preset cannot store")
    try:
        return converter(text)
    except ValueError as error:
        raise PresetError(f"{name} needs a number, got '{text}'") from error


def format_preset(pairs: list[Pair]) -> str:
    preset_text = "\n".join([PRESET_HEADER, *(f"{name}={text}" for name, text in pairs)]) + "\n"
    # set_param passes len(str) as the length, which is the byte count only for ASCII.
    if not preset_text.isascii() or len(preset_text) > MAX_PRESET_BYTES:
        raise PresetError(f"the preset must be ASCII text of at most {MAX_PRESET_BYTES} bytes")
    return preset_text


def read_ffs_text(cam: xiapi.Camera) -> str | None:
    """Returns the content of the preset file, or None when no preset is stored."""
    select_ffs_file(cam)
    # Families report a missing file differently: size 0, error 113, or (MX)
    # error 12 when an empty file is read. So check the size before reading.
    try:
        if cam.get_ffs_file_size() == 0:
            return None
        data = cam.get_read_file_ffs(buffer_size=MAX_PRESET_BYTES + 1)
    except xiapi.Xi_error as error:
        if error.status == XI_FFS_FILE_NOT_FOUND:
            return None
        raise
    preset_text = data.split(b"\0", 1)[0].decode("ascii", errors="replace")
    if not preset_text.startswith(PRESET_HEADER):
        raise PresetError(f"FFS file '{FFS_FILE_NAME}' holds other data; 'save' would overwrite it")
    return preset_text


def select_ffs_file(cam: xiapi.Camera) -> None:
    try:
        cam.set_ffs_file_name(FFS_FILE_NAME)
    except xiapi.Xi_error as error:
        if error.status in XI_FFS_UNSUPPORTED_CODES:
            raise PresetError(f"this camera does not support the flash file system (FFS): {error}") from error
        raise


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
    select_ffs_file(cam)
    cam.set_write_file_ffs(preset_text)
    if read_ffs_text(cam) != preset_text:
        raise PresetError(f"verification failed: '{FFS_FILE_NAME}' does not contain the saved preset")
    print(f"Preset saved to FFS file '{FFS_FILE_NAME}'")


def parse_preset(preset_text: str) -> list[Pair]:
    lines = (line.strip() for line in preset_text.splitlines())
    return [parse_pair(line) for line in lines if line and not line.startswith("#")]


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
    print(f"Opening camera {serial or 'index 0'}")
    if serial:
        cam.open_device_by_SN(serial)
    else:
        cam.open_device()
    try:
        yield cam
    finally:
        cam.close_device()


def main() -> int:
    parser = argparse.ArgumentParser(description="Save, show or load a parameter preset in the camera FFS.")
    parser.add_argument("-s", "--serial", help="serial number of the camera; default is the first camera")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("save", help="apply name=value pairs and store them").add_argument("pairs", nargs="+")
    commands.add_parser("show", help="print the stored preset (default)")
    commands.add_parser("load", help="apply the stored preset to the camera")
    args = parser.parse_args()

    try:
        pairs: list[Pair] = []
        if args.command == "save":
            # Validate everything before the camera is opened.
            pairs = [parse_pair(item) for item in args.pairs]
            for name, text in pairs:
                parse_value(name, text)
            format_preset(pairs)

        with open_camera(args.serial) as cam:
            if args.command == "save":
                save_preset(cam, pairs)
            elif args.command == "load":
                load_preset(cam)
            else:
                show_preset(cam)
    except PresetError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except xiapi.Xi_error as error:
        print(f"Error: xiAPI call failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
