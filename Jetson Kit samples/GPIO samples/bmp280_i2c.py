'''
Docstring for Jetson Kit samples.GPIO samples.bmp280_i2c


Connect:
Vin to gpio header pin 5 (3V3)
GND to gpio header pin 6 (GND)
SCK to gpio header pin 27 (I2C1 SCL)
SDI to gpio header pin 28 (I2C1 SDA)

Adafruit board:
https://learn.adafruit.com/adafruit-bmp280-barometric-pressure-plus-temperature-sensor-breakout

BMP280 Datasheet:
https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp280-ds001.pdf

Pylibi2c documentation: 
https://github.com/amaork/libi2c
'''

import time
import sys
import pylibi2c
from numpy import double

#Using device i2c-7, which is mapped to I2C1 on Orin NX, GPIO header pins 27 (SCL) and 28 (SDA)
I2C_BUS = "/dev/i2c-7"   
BMP280_ADDR = 0x77  # 0x76 if SDO=GND, 0x77 if SDO=VDDIO (datasheet chapter 5.2)


def read_register(address, n=1) -> bytes:
    # write register address, repeated-start read
    return i2c.read(address & 0xFF, n)

def write_register(address, value: int) -> None:
    # Write a single byte to a register
    i2c.write(address & 0xFF, bytes([value & 0xFF]))

def compensate_temperature(adc_T,dig_T1,dig_T2,dig_T3):
    #From chapter 3.11.3 of the datasheet
    var1 = (((int(adc_T)>>3) - (int(dig_T1) <<1))* int(dig_T2)) >> 11
    var2 = (((((int(adc_T)>>4) - int(dig_T1)) * ((int(adc_T)>>4) - int(dig_T1))) >> 12 ) * int(dig_T3)) >>14

    t_fine = var1 + var2
    temp = (t_fine * 5 + 128) >> 8
    return (double (temp) / 100.0)

print("BMP280 I2C sample")

# Initialize I2C device
i2c = pylibi2c.I2CDevice(I2C_BUS, BMP280_ADDR, iaddr_bytes=1)

# Read chip ID
chip_id = read_register(0xD0, 1)[0] # Datacheet chapter 4.3.1
print("BMP280 chip id:", hex(chip_id))  # expected 0x58

if chip_id != 0x58:
    i2c.close()
    sys.exit("BMP280 not found on I2C bus")

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
    i2c.close()
