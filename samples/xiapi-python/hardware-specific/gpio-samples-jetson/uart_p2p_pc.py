"""Run the PC side of a peer-to-peer UART PING/PONG sample.

The PC sends a PING message to the Jetson, and the Jetson replies with PONG.
UART0 is used to demonstrate RTS#/CTS# flow control.
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

PORT = "/dev/ttyUSB0"
BAUDRATE = 115_200
ITERATION_COUNT = 1_000
SERIAL_TIMEOUT_S = 2.0
LOOP_DELAY_S = 0.5


def main() -> int:
    serial_port = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=SERIAL_TIMEOUT_S,
        write_timeout=SERIAL_TIMEOUT_S,
        rtscts=True,
    )

    try:
        serial_port.reset_input_buffer()
        serial_port.reset_output_buffer()

        for i in range(ITERATION_COUNT):
            serial_port.write(b"PING\n")
            serial_port.flush()

            read_data = serial_port.read_until(b"\n")
            if read_data.strip() == b"PONG":
                print(f"Iteration {i}:\nSuccess: Sent PING, received PONG")
            elif read_data == b"":
                print(f"Iteration {i}:\nError: Received no data")
            else:
                decoded_data = read_data.decode(errors="replace").strip()
                print(f"Iteration {i}:\nError: Received: {decoded_data}")

            time.sleep(LOOP_DELAY_S)
    except serial.SerialTimeoutException as error:
        print("Timeout occurred. Aborting!")
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        serial_port.rtscts = False
        serial_port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
