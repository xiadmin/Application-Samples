"""
Docstring for uart_loopback

Connect:
Gpio header pin 20 (UART1 TX) to gpio header pin 21 (UART1 RX)

PySerial documentation:
https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial
"""

import time
import serial

print("UART Loopback sample")

# Using device ttyHS1, which is mapped to UART1 on Orin NX, GPIO header pins 20 (TX) and 21 (RX)
serial_port = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

# Wait a second to let the port initialize
time.sleep(1)

try:
    for i in range(100):
        #Sending data
        send_data = f'Loopback iteration {i}\n'.encode()
        serial_port.write(send_data)
        
        time.sleep(0.1)
        
        #Receiving data
        read_data = serial_port.read_until(b'\n')
        print(f"Received: {read_data.decode().strip()}")
        
        time.sleep(0.5)

except serial.SerialTimeoutException as exception_error:
    print("Error occurred. Aborting!")
    print("Error: " + str(exception_error))

except KeyboardInterrupt:
    print("Aborting!")

finally:
    serial_port.close()