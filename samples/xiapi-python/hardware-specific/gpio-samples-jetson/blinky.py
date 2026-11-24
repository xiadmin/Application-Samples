"""Blink the outer green LED on the Jetson Kit with libgpiod.

Outer green LED (GPIO11) is used in the sample to avoid the need for
additional hardware setup. Abort the program with Ctrl+C.

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path


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


try:
    import gpiod
    from gpiod.line import Direction, Drive, Value
except ImportError:
    if not install_requirements("gpiod"):
        sys.exit(1)
    import gpiod
    from gpiod.line import Direction, Drive, Value

# Outer green LED, GPIO11/PQ.06 on Orin NX: gpiochip0 line offset 106.
DEVICE = "/dev/gpiochip0"
OFFSET = 106
BLINK_COUNT = 100
BLINK_DELAY_S = 0.4


def main() -> int:
    request = gpiod.request_lines(
        DEVICE,
        consumer="simple-output",
        config={
            OFFSET: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                drive=Drive.PUSH_PULL,
                active_low=True,
                output_value=Value.INACTIVE,
            )
        },
    )

    try:
        print("Blinking outer LED GREEN.")
        for _ in range(BLINK_COUNT):
            request.set_value(OFFSET, Value.ACTIVE)
            time.sleep(BLINK_DELAY_S)
            request.set_value(OFFSET, Value.INACTIVE)
            time.sleep(BLINK_DELAY_S)
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        request.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
