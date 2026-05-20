"""
This is a sample code for demonstrating UART communication on the Jetson Kit.
A UART loopback test is performed by connecting the TX and RX pins together.
PySerial is used for UART communication.

Workflow:
- Initialize UART communication
- In a loop:
-- Send data through TX pin
-- Read data from RX pin

Abort the program with Ctrl+C.

UART1 connection:
Gpio header pin 20 (UART1 TX) to gpio header pin 21 (UART1 RX)

UART0 connection:
Gpio header pin 18 (UART0 TX) to gpio header pin 19 (UART0 RX)

PySerial documentation:
https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial
"""

import time
import serial

print("UART Loopback sample")

# Using device ttyHS1, which is mapped to UART1 on Orin NX, GPIO header pins 20 (TX) and 21 (RX)
# Alternately using device ttyTHS3, which is mapped to UART0 on Orin NX, GPIO header pins 18 (TX) and 19 (RX)
serial_port = serial.Serial(
    port="/dev/ttyTHS1", #For UART1
    #port="/dev/ttyTHS3", #For UART0  
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    rtscts=False 
)

# Wait a second to let the port initialize
time.sleep(1)

try:
    # Clear buffers before starting
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()

    for i in range(100):
        #Sending data
        send_data = f'Loopback iteration {i}\n'.encode()
        serial_port.write(send_data)
        
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