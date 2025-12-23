import time
import pylibi2c

I2C_BUS = "/dev/i2c-7"   # adjust for your platform (e.g. /dev/i2c-0, /dev/i2c-1)
BMP280_ADDR = 0x77       # 0x76 if SDO=GND, 0x77 if SDO=VDDIO (per datasheet)

i2c = pylibi2c.I2CDevice(I2C_BUS, BMP280_ADDR, iaddr_bytes=1)


def read_register(address, n=1) -> bytes:
    # Combined transaction (write register address, repeated-start read)
    return i2c.ioctl_read(address & 0xFF, n)

def write_register(address, value: int) -> None:
    # Write a single byte to a register
    i2c.write(address & 0xFF, bytes([value & 0xFF]))

chip_id = read_register(0xD0, 1)[0]
print("BMP280 chip id:", hex(chip_id))  # expected 0x58


OSRS_T = 0b001       # x1
OSRS_P = 0b000       # skipped
MODE_FORCED = 0b01   # forced mode (one-shot)

ctrl_meas_forced = (OSRS_T << 5) | (OSRS_P << 2) | MODE_FORCED

for _ in range(100):
    # Trigger one temperature conversion
    write_register(0xF4, ctrl_meas_forced)
    time.sleep(0.01) 
    msb, lsb, xlsb = read_register(0xFA, 3)
    # ut[19:0] = msb[7:0] lsb[7:0] xlsb[7:4]
    value = (msb << 12) | (lsb << 4) | (xlsb >> 4)
    
    print(f"Raw temperature reading: {value} DN")
    time.sleep(0.5)    


i2c.close()
