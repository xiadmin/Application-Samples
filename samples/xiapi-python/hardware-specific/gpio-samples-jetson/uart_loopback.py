"""Run a UART loopback test on the Jetson Kit with PySerial.

Connect TX and RX pins together. Abort the program with Ctrl+C.

UART1 connection:
GPIO header pin 20 (UART1 TX) to GPIO header pin 21 (UART1 RX)

UART0 connection:
GPIO header pin 18 (UART0 TX) to GPIO header pin 19 (UART0 RX)

PySerial documentation:
https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial
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
    import serial
except ImportError:
    if not install_requirements("pyserial"):
        sys.exit(1)
    import serial

PORT = "/dev/ttyTHS1"  # Use "/dev/ttyTHS3" for UART0.
BAUDRATE = 115_200
LOOP_COUNT = 100
LOOP_DELAY_S = 0.5
PORT_INITIALIZE_DELAY_S = 1.0


def main() -> int:
    print("UART Loopback sample")
    serial_port = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        rtscts=False,
    )

    time.sleep(PORT_INITIALIZE_DELAY_S)

    try:
        serial_port.reset_input_buffer()
        serial_port.reset_output_buffer()

        for i in range(LOOP_COUNT):
            send_data = f"Loopback iteration {i}\n".encode()
            serial_port.write(send_data)

            read_data = serial_port.read_until(b"\n")
            print(f"Received: {read_data.decode().strip()}")
            time.sleep(LOOP_DELAY_S)
    except serial.SerialTimeoutException as error:
        print("Error occurred. Aborting!")
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        serial_port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
