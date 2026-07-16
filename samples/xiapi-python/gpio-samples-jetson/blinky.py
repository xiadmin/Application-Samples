"""Blink the outer green LED on the Jetson Kit with libgpiod.

Outer green LED (GPIO11) is used in the sample to avoid the need for
additional hardware setup. Abort the program with Ctrl+C.

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import time

import gpiod
from gpiod.line import Direction, Drive, Value

# Outer green LED, GPIO11/PQ.06 on Orin NX: gpiochip0 line offset 106.
DEVICE = "/dev/gpiochip0"
OFFSET = 106
BLINK_COUNT = 100
BLINK_DELAY_S = 0.4


def main() -> int:
    request = gpiod.request_lines(
        DEVICE,
        consumer="simple-output",
        config={
            OFFSET: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                drive=Drive.PUSH_PULL,
                active_low=True,
                output_value=Value.INACTIVE,
            )
        },
    )

    try:
        print("Blinking outer LED GREEN.")
        for _ in range(BLINK_COUNT):
            request.set_value(OFFSET, Value.ACTIVE)
            time.sleep(BLINK_DELAY_S)
            request.set_value(OFFSET, Value.INACTIVE)
            time.sleep(BLINK_DELAY_S)
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        request.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
