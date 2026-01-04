"""
This is a sample code for demonstrating SPI communication on the Jetson Kit.
BMP280 sensor embedded in an Adafruit board is used as communication target.
PySpidev is used for SPI communication.

Workflow:
- Initialize SPI communication
- Read BMP280 chip ID
- Read calibration parameters from BMP280
- Trigger and read temperature measurement in a loop

Connect:
Vin to gpio header pin 5 (3V3)
GND to gpio header pin 6 (GND)
SCK to gpio header pin 23 (CLK)
SDO to gpio header pin 22 (MISO)
SDI to gpio header pin 24 (MOSI)
CS to gpio header pin 25 (CS0)

Adafruit board:
https://learn.adafruit.com/adafruit-bmp280-barometric-pressure-plus-temperature-sensor-breakout

BMP280 Datasheet:
https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp280-ds001.pdf

PySpidev documentation: 
https://github.com/doceme/py-spidev
"""

import time
import sys
import spidev
from numpy import double


# Using device spidev0.0, which is the SPI0 on Orin NX, chip select 0
# Set to 3V3 logic

BUS = 0
CS  = 0  

def read_register(address, bytes_n=1):
    # bit7 = 1 for read; address is in bits6:0
    resp = spi.xfer2([address | 0x80] + [0x00] * bytes_n)
    return resp[1:]

def write_register(address, value):
    # bit7 = 0 for write
    spi.xfer2([address & 0x7F, value & 0xFF])


def compensate_temperature(adc_T,dig_T1,dig_T2,dig_T3):
    #From chapter 3.11.3 of the datasheet
    var1 = (((int(adc_T)>>3) - (int(dig_T1) <<1))* int(dig_T2)) >> 11
    var2 = (((((int(adc_T)>>4) - int(dig_T1)) * ((int(adc_T)>>4) - int(dig_T1))) >> 12 ) * int(dig_T3)) >>14

    t_fine = var1 + var2
    temp = (t_fine * 5 + 128) >> 8
    return (double (temp) / 100.0)

print("BMP280 SPI sample")

# Initialize SpiDev device
spi = spidev.SpiDev()
spi.open(BUS, CS)
spi.mode = 0  # BMP280 supports SPI mode 0 and 3
spi.max_speed_hz = 1_000_000

# Read chip ID
chip_id = read_register(0xD0, 1)[0] # Datacheet chapter 4.3.1
print("BMP280 chip id:", hex(chip_id))  # expected 0x58

if chip_id != 0x58:
    spi.close()
    sys.exit("BMP280 not found on the SPI bus")

# Reading compensation parameters, Datasheet chapter 3.11.2
dig_T1 = int.from_bytes(read_register(0x88, 2), 'little')
dig_T2 = int.from_bytes(read_register(0x8A, 2), 'little', signed=True)
dig_T3 = int.from_bytes(read_register(0x8C, 2), 'little', signed=True)

# Datasheet chapter 4.3.4
OSRS_T = 0b001       # temperature oversampling x1
OSRS_P = 0b000       # pressure oversampling x0, measurement skipped, 
MODE_FORCED = 0b01   # forced mode (one-shot)

ctrl_meas_forced = (OSRS_T << 5) | (OSRS_P << 2) | MODE_FORCED


try:
    for _ in range(100):

        # Trigger one temperature measurement
        write_register(0xF4, ctrl_meas_forced)
        
        # Wait for measurement to complete
        for i in range(10):
            status = read_register(0xF3, 1)[0] # datasheet chapter 4.3.3
            if (status & 0b00001000) == 0:
                break
            time.sleep(0.01) 

        if i == 9:
            raise Exception("Temperature measurement timed out")

        # Reading raw temperature data
        msb, lsb, xlsb = read_register(0xFA, 3) # Datasheet chapter 4.3.7
        raw_value = (msb << 12) | (lsb << 4) | (xlsb >> 4) # temp[19:0] = msb[7:0] lsb[7:0] xlsb[7:4]
        
        #Cauculated calibrated temperature
        temp_value = compensate_temperature(raw_value, dig_T1, dig_T2, dig_T3)
        print(f"Temperature reading: {temp_value:.2f} °C")

        time.sleep(0.5)

except Exception as exception_error:
    print("Error occurred. Aborting!")
    print("Error: " + str(exception_error))    

except KeyboardInterrupt:
    print("Aborting!")

finally:
    spi.close()