"""Run the Jetson side of a peer-to-peer UART PING/PONG sample.

The PC sends a PING message to the Jetson, and the Jetson replies with PONG.
UART0 is used to demonstrate RTS#/CTS# flow control.
"""

import serial

PORT = "/dev/ttyTHS3"
BAUDRATE = 115_200
ITERATION_COUNT = 1_000
SERIAL_TIMEOUT_S = 2.0


def main() -> int:
    serial_port = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=SERIAL_TIMEOUT_S,
        write_timeout=SERIAL_TIMEOUT_S,
        rtscts=True,
    )

    try:
        serial_port.reset_input_buffer()
        serial_port.reset_output_buffer()

        for i in range(ITERATION_COUNT):
            read_data = serial_port.read_until(b"\n")

            if read_data.strip() == b"PING":
                print(f"Iteration {i}:\nSuccess: Received PING, sending PONG")
                serial_port.write(b"PONG\n")
                serial_port.flush()
            elif read_data == b"":
                print(f"Iteration {i}:\nError: Received no data.")
            else:
                decoded_data = read_data.decode(errors="replace").strip()
                print(f"Iteration {i}:\nError: Received: {decoded_data}")
    except serial.SerialTimeoutException as error:
        print("Timeout occurred. Aborting!")
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        serial_port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
