"""Read BMP280 temperature over I2C on the Jetson Kit.

BMP280 sensor embedded in an Adafruit board is used as communication target.
smbus2 is used for I2C communication. Abort the program with Ctrl+C.
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
    from smbus2 import SMBus
except ImportError:
    if not install_requirements("smbus2"):
        sys.exit(1)
    from smbus2 import SMBus

# Device i2c-7 is mapped to I2C1 on Orin NX GPIO header pins 27 (SCL) and 28 (SDA).
I2C_BUS = "/dev/i2c-7"
BMP280_ADDR = 0x77  # 0x76 if SDO=GND, 0x77 if SDO=VDDIO.
EXPECTED_CHIP_ID = 0x58
TEMPERATURE_READ_COUNT = 100
TEMPERATURE_READ_DELAY_S = 0.5
MEASUREMENT_RETRY_COUNT = 10
MEASUREMENT_RETRY_DELAY_S = 0.01
OSRS_T = 0b001
OSRS_P = 0b000
MODE_FORCED = 0b01
CTRL_MEAS_FORCED = (OSRS_T << 5) | (OSRS_P << 2) | MODE_FORCED


def read_register(i2c: SMBus, address: int, byte_count: int = 1) -> bytes:
    # The BMP280 auto-increments the register address, so one block read
    # returns consecutive registers.
    return bytes(i2c.read_i2c_block_data(BMP280_ADDR, address & 0xFF, byte_count))


def write_register(i2c: SMBus, address: int, value: int) -> None:
    i2c.write_byte_data(BMP280_ADDR, address & 0xFF, value & 0xFF)


def compensate_temperature(adc_temperature: int, dig_t1: int, dig_t2: int, dig_t3: int) -> float:
    # Formula from BMP280 datasheet chapter 3.11.3.
    var1 = (((adc_temperature >> 3) - (dig_t1 << 1)) * dig_t2) >> 11
    var2 = (((((adc_temperature >> 4) - dig_t1) * ((adc_temperature >> 4) - dig_t1)) >> 12) * dig_t3) >> 14
    t_fine = var1 + var2
    temp = (t_fine * 5 + 128) >> 8
    return temp / 100.0


def read_calibration(i2c: SMBus) -> tuple[int, int, int]:
    dig_t1 = int.from_bytes(read_register(i2c, 0x88, 2), "little")
    dig_t2 = int.from_bytes(read_register(i2c, 0x8A, 2), "little", signed=True)
    dig_t3 = int.from_bytes(read_register(i2c, 0x8C, 2), "little", signed=True)
    return dig_t1, dig_t2, dig_t3


def wait_for_measurement(i2c: SMBus) -> None:
    for i in range(MEASUREMENT_RETRY_COUNT):
        status = read_register(i2c, 0xF3, 1)[0]
        if (status & 0b00001000) == 0:
            return
        time.sleep(MEASUREMENT_RETRY_DELAY_S)
    raise TimeoutError("Temperature measurement timed out")


def main() -> int:
    print("BMP280 I2C sample")
    try:
        with SMBus(I2C_BUS) as i2c:
            chip_id = read_register(i2c, 0xD0, 1)[0]
            print("BMP280 chip id:", hex(chip_id))
            if chip_id != EXPECTED_CHIP_ID:
                print("BMP280 not found on I2C bus", file=sys.stderr)
                return 1

            dig_t1, dig_t2, dig_t3 = read_calibration(i2c)

            for _ in range(TEMPERATURE_READ_COUNT):
                write_register(i2c, 0xF4, CTRL_MEAS_FORCED)
                wait_for_measurement(i2c)
                msb, lsb, xlsb = read_register(i2c, 0xFA, 3)
                raw_value = (msb << 12) | (lsb << 4) | (xlsb >> 4)
                temp_value = compensate_temperature(raw_value, dig_t1, dig_t2, dig_t3)
                print(f"Temperature reading: {temp_value:.2f} °C")
                time.sleep(TEMPERATURE_READ_DELAY_S)
    # OSError covers a missing bus device, missing permissions and a sensor
    # that does not acknowledge its address.
    except (OSError, TimeoutError) as error:
        print(f"Error: {error}. Aborting!", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Aborting!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
