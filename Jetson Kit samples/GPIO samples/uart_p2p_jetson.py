"""
This is a sample code demonstrating Peer-to-Peer UART communication between a Linux PC and the Jetson kit.
The PC sends a "PING" message to the Jetson, and the Jetson replies with a "PONG" message. This process is repeated in a loop.
UART0 is used to demonstrate the functionality of RTS#/CTS# flow control.
PySerial is used for UART communication. 

Assumptions taken:
- The PC is running Linux, Ubuntu.
- 5 pin 1V8 USB to TTL UART converter is used for communication.   

Workflow:
- Initialize UART communication
- In a loop:
-- Receive "PING" message from PC
-- Send "PONG" message to PC 

Connect: 

BLACK wire (GND) - GPIO header pin 6 (GND)
ORANGE wire (TX) - GPIO header pin 18 (RX)
YELLOW wire (RX) - GPIO header pin 19 (TX)
GREEN wire (RTS#) - GPIO header pin 17 (CTS#)
BROWN wire (CTS#) - GPIO header pin 16 (RTS#)

USB to UART Converter:
https://ftdichip.com/products/ttl-232rg-vreg1v8-we/
Datasheet:
https://ftdichip.com/wp-content/uploads/2023/07/DS_TTL-232RG_CABLES.pdf

"""

import time
import serial
   
# Using UART0, which has RTS#/CTS# flow control
serial_port= serial.Serial(
    port = "/dev/ttyTHS3",  # For UART0
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=2.0,
    write_timeout=2.0,
    rtscts=True #Enabling Flow control to demonstrate funtionality
)


try:
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()

    for i in range(1000):
        read_data = serial_port.read_until(b'\n')  

        if read_data.strip() == b"PING":
            print(f"Iteration {i}: \nSuccess: Received PING, sending PONG")
            
            serial_port.write(b"PONG\n")
            serial_port.flush()
        
        elif read_data == b'':
            print(f"Iteration {i}: \nError: Received no data.")
        
        else:
            print(f"Iteration {i}: \n Error: Received: " + read_data.decode().strip())


except serial.SerialTimeoutException as exception_error:
    print("Timeout occurred. Aborting!")
    print("Error: " + str(exception_error))

except KeyboardInterrupt:
    print("Aborting!")

finally:
    serial_port.close()

