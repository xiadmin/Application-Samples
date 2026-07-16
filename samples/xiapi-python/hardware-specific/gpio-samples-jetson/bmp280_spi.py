"""Read BMP280 temperature over SPI on the Jetson Kit.

BMP280 sensor embedded in an Adafruit board is used as communication target.
PySpidev is used for SPI communication. Abort the program with Ctrl+C.
"""

import sys
import time

import spidev

BUS = 0
CHIP_SELECT = 0
EXPECTED_CHIP_ID = 0x58
SPI_MODE = 0
SPI_MAX_SPEED_HZ = 1_000_000
TEMPERATURE_READ_COUNT = 100
TEMPERATURE_READ_DELAY_S = 0.5
MEASUREMENT_RETRY_COUNT = 10
MEASUREMENT_RETRY_DELAY_S = 0.01
OSRS_T = 0b001
OSRS_P = 0b000
MODE_FORCED = 0b01
CTRL_MEAS_FORCED = (OSRS_T << 5) | (OSRS_P << 2) | MODE_FORCED


def read_register(spi, address: int, byte_count: int = 1) -> list[int]:
    response = spi.xfer2([address | 0x80] + [0x00] * byte_count)
    return response[1:]


def write_register(spi, address: int, value: int) -> None:
    spi.xfer2([address & 0x7F, value & 0xFF])


def compensate_temperature(adc_temperature: int, dig_t1: int, dig_t2: int, dig_t3: int) -> float:
    # Formula from BMP280 datasheet chapter 3.11.3.
    var1 = (((adc_temperature >> 3) - (dig_t1 << 1)) * dig_t2) >> 11
    var2 = (((((adc_temperature >> 4) - dig_t1) * ((adc_temperature >> 4) - dig_t1)) >> 12) * dig_t3) >> 14
    t_fine = var1 + var2
    temp = (t_fine * 5 + 128) >> 8
    return temp / 100.0


def read_calibration(spi) -> tuple[int, int, int]:
    dig_t1 = int.from_bytes(read_register(spi, 0x88, 2), "little")
    dig_t2 = int.from_bytes(read_register(spi, 0x8A, 2), "little", signed=True)
    dig_t3 = int.from_bytes(read_register(spi, 0x8C, 2), "little", signed=True)
    return dig_t1, dig_t2, dig_t3


def wait_for_measurement(spi) -> None:
    for _ in range(MEASUREMENT_RETRY_COUNT):
        status = read_register(spi, 0xF3, 1)[0]
        if (status & 0b00001000) == 0:
            return
        time.sleep(MEASUREMENT_RETRY_DELAY_S)
    raise TimeoutError("Temperature measurement timed out")


def main() -> int:
    print("BMP280 SPI sample")
    spi = spidev.SpiDev()
    spi.open(BUS, CHIP_SELECT)
    spi.mode = SPI_MODE
    spi.max_speed_hz = SPI_MAX_SPEED_HZ

    try:
        chip_id = read_register(spi, 0xD0, 1)[0]
        print("BMP280 chip id:", hex(chip_id))
        if chip_id != EXPECTED_CHIP_ID:
            print("BMP280 not found on the SPI bus", file=sys.stderr)
            return 1

        dig_t1, dig_t2, dig_t3 = read_calibration(spi)

        for _ in range(TEMPERATURE_READ_COUNT):
            write_register(spi, 0xF4, CTRL_MEAS_FORCED)
            wait_for_measurement(spi)
            msb, lsb, xlsb = read_register(spi, 0xFA, 3)
            raw_value = (msb << 12) | (lsb << 4) | (xlsb >> 4)
            temp_value = compensate_temperature(raw_value, dig_t1, dig_t2, dig_t3)
            print(f"Temperature reading: {temp_value:.2f} °C")
            time.sleep(TEMPERATURE_READ_DELAY_S)
    except (OSError, TimeoutError) as error:
        print("Error occurred. Aborting!")
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        spi.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
