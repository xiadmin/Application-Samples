"""
Docstring for uart_loopback
"""

import time
import serial

print("UART Loopback sample")

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
    for i in range(10):
        
        send_data = f'Loopback iteration {i}\n'.encode()
        serial_port.write(send_data)
        time.sleep(0.1)
        read_data = serial_port.read_until(b'\n')
        print(f"Received: {read_data.decode().strip()}")
        time.sleep(0.5)
        
    
except KeyboardInterrupt:
    print("Exiting Program")

except Exception as exception_error:
    print("Error occurred. Exiting Program")
    print("Error: " + str(exception_error))

finally:
    serial_port.close()