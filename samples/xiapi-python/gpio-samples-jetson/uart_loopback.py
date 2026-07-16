"""Run a UART loopback test on the Jetson Kit with PySerial.

Connect TX and RX pins together. Abort the program with Ctrl+C.

UART1 connection:
GPIO header pin 20 (UART1 TX) to GPIO header pin 21 (UART1 RX)

UART0 connection:
GPIO header pin 18 (UART0 TX) to GPIO header pin 19 (UART0 RX)

PySerial documentation:
https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.Serial
"""

import time

import serial

PORT = "/dev/ttyTHS1"  # Use "/dev/ttyTHS3" for UART0.
BAUDRATE = 115_200
LOOP_COUNT = 100
LOOP_DELAY_S = 0.5
PORT_INITIALIZE_DELAY_S = 1.0


def main() -> int:
    print("UART Loopback sample")
    serial_port = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        rtscts=False,
    )

    time.sleep(PORT_INITIALIZE_DELAY_S)

    try:
        serial_port.reset_input_buffer()
        serial_port.reset_output_buffer()

        for i in range(LOOP_COUNT):
            send_data = f"Loopback iteration {i}\n".encode()
            serial_port.write(send_data)

            read_data = serial_port.read_until(b"\n")
            print(f"Received: {read_data.decode().strip()}")
            time.sleep(LOOP_DELAY_S)
    except serial.SerialTimeoutException as error:
        print("Error occurred. Aborting!")
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        serial_port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
