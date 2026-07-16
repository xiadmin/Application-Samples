"""Read BMP280 temperature over I2C on the Jetson Kit.

BMP280 sensor embedded in an Adafruit board is used as communication target.
Pylibi2c is used for I2C communication. Abort the program with Ctrl+C.
"""

import sys
import time

import pylibi2c

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


def read_register(i2c, address: int, byte_count: int = 1) -> bytes:
    return i2c.read(address & 0xFF, byte_count)


def write_register(i2c, address: int, value: int) -> None:
    i2c.write(address & 0xFF, bytes([value & 0xFF]))


def compensate_temperature(adc_temperature: int, dig_t1: int, dig_t2: int, dig_t3: int) -> float:
    # Formula from BMP280 datasheet chapter 3.11.3.
    var1 = (((adc_temperature >> 3) - (dig_t1 << 1)) * dig_t2) >> 11
    var2 = (((((adc_temperature >> 4) - dig_t1) * ((adc_temperature >> 4) - dig_t1)) >> 12) * dig_t3) >> 14
    t_fine = var1 + var2
    temp = (t_fine * 5 + 128) >> 8
    return temp / 100.0


def read_calibration(i2c) -> tuple[int, int, int]:
    dig_t1 = int.from_bytes(read_register(i2c, 0x88, 2), "little")
    dig_t2 = int.from_bytes(read_register(i2c, 0x8A, 2), "little", signed=True)
    dig_t3 = int.from_bytes(read_register(i2c, 0x8C, 2), "little", signed=True)
    return dig_t1, dig_t2, dig_t3


def wait_for_measurement(i2c) -> None:
    for i in range(MEASUREMENT_RETRY_COUNT):
        status = read_register(i2c, 0xF3, 1)[0]
        if (status & 0b00001000) == 0:
            return
        time.sleep(MEASUREMENT_RETRY_DELAY_S)
    raise TimeoutError("Temperature measurement timed out")


def main() -> int:
    print("BMP280 I2C sample")
    i2c = pylibi2c.I2CDevice(I2C_BUS, BMP280_ADDR, iaddr_bytes=1)

    try:
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
    except Exception as error:
        print("Error occurred. Aborting!")
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        i2c.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
