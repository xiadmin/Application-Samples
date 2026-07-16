"""Read the outer button state on the Jetson Kit with libgpiod.

Outer button (GPIO10) is used in the sample to avoid the need for additional
hardware setup. Abort the program with Ctrl+C.

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import time

import gpiod
from gpiod.line import Direction

# Outer button, GPIO10/PEE.02 on Orin NX: gpiochip1 line offset 25.
DEVICE = "/dev/gpiochip1"
OFFSET = 25
READ_COUNT = 100
READ_DELAY_S = 0.5


def main() -> int:
    request = gpiod.request_lines(
        DEVICE,
        consumer="simple-input",
        config={
            OFFSET: gpiod.LineSettings(
                direction=Direction.INPUT,
                active_low=True,
            )
        },
    )

    try:
        for _ in range(READ_COUNT):
            value = request.get_value(OFFSET)
            print(f"Outer button value: {value.value}")
            time.sleep(READ_DELAY_S)
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        request.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
