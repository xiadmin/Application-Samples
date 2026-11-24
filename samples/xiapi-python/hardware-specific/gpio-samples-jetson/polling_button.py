"""Toggle the outer green LED from the outer button using polling debounce.

Outer button (GPIO10) and outer green LED (GPIO11) are used in the sample.
Libgpiod is used for GPIO control. Abort the program with Ctrl+C.

Sample is based on:
https://docs.arduino.cc/built-in-examples/digital/Debounce/

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

# Outer button, GPIO10/PEE.02 on Orin NX: gpiochip1 line offset 25.
BUTTON_DEVICE = "/dev/gpiochip1"
BUTTON_OFFSET = 25

# Outer green LED, GPIO11/PQ.06 on Orin NX: gpiochip0 line offset 106.
LED_DEVICE = "/dev/gpiochip0"
LED_OFFSET = 106

DEBOUNCE_MS = 50
NS_PER_MS = 1_000_000


def monotonic_ms() -> int:
    return time.monotonic_ns() // NS_PER_MS


def main() -> int:
    button_settings = gpiod.LineSettings(
        direction=Direction.INPUT,
        active_low=True,
    )
    led_settings = gpiod.LineSettings(
        direction=Direction.OUTPUT,
        drive=Drive.PUSH_PULL,
        active_low=True,
        output_value=Value.INACTIVE,
    )

    button_request = gpiod.request_lines(
        BUTTON_DEVICE,
        consumer="button-input",
        config={BUTTON_OFFSET: button_settings},
    )
    led_request = gpiod.request_lines(
        LED_DEVICE,
        consumer="led-output",
        config={LED_OFFSET: led_settings},
    )

    try:
        print("Press the button to toggle the outer green LED")
        last_button_value = Value.INACTIVE
        is_led_active = False
        button_state = Value.INACTIVE
        last_debounce_time = monotonic_ms()

        while True:
            button_value = button_request.get_value(BUTTON_OFFSET)
            if button_value is not last_button_value:
                last_debounce_time = monotonic_ms()

            if monotonic_ms() - last_debounce_time > DEBOUNCE_MS:
                if button_value != button_state:
                    button_state = button_value
                    if button_state is Value.ACTIVE:
                        print("Button pressed, toggling LED")
                        is_led_active = not is_led_active

            led_value = Value.ACTIVE if is_led_active else Value.INACTIVE
            led_request.set_value(LED_OFFSET, led_value)
            last_button_value = button_value
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        button_request.release()
        led_request.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
