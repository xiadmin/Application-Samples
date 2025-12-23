"""
TODO: Lorem ipsum
"""


import time
import spidev

BUS = 0
CS  = 0  # adjust to your /dev/spidevX.Y

spi = spidev.SpiDev()
spi.open(BUS, CS)
spi.mode = 0  # BMP280 supports SPI mode 0 and 3 per datasheet
spi.max_speed_hz = 1_000_000

def read_register(address, bytes_n=1):
    # bit7 = 1 for read; address is in bits6:0
    resp = spi.xfer2([address | 0x80] + [0x00] * bytes_n)
    return resp[1:]

def write_register(address, value):
    # bit7 = 0 for write
    spi.xfer2([address & 0x7F, value & 0xFF])


chip_id = read_register(0xD0, 1)[0]
print("BMP280 chip id:", hex(chip_id))  # expected 0x58


for i in range(100):
    OSRS_T = 0b001  # x1
    OSRS_P = 0b000  # skipped
    MODE_FORCED = 0b01

    ctrl_meas_forced = (OSRS_T << 5) | (OSRS_P << 2) | MODE_FORCED
    write_register(0xF4, ctrl_meas_forced)

    msb, lsb, xlsb = read_register(0xFA, 3)
    # ut[19:0] = msb[7:0] lsb[7:0] xlsb[7:4]
    value = (msb << 12) | (lsb << 4) | (xlsb >> 4)
    print(f"Raw temperature reading: {value} DN")
    time.sleep(0.5)

spi.close()