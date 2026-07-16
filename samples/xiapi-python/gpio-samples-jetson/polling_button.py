"""Toggle the outer green LED from the outer button using polling debounce.

Outer button (GPIO10) and outer green LED (GPIO11) are used in the sample.
Libgpiod is used for GPIO control. Abort the program with Ctrl+C.

Sample is based on:
https://docs.arduino.cc/built-in-examples/digital/Debounce/

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import time

import gpiod
from gpiod.line import Direction, Drive, Value

# Outer button, GPIO10/PEE.02 on Orin NX: gpiochip1 line offset 25.
BUTTON_DEVICE = "/dev/gpiochip1"
BUTTON_OFFSET = 25

# Outer green LED, GPIO11/PQ.06 on Orin NX: gpiochip0 line offset 106.
LED_DEVICE = "/dev/gpiochip0"
LED_OFFSET = 106

DEBOUNCE_MS = 50
NS_PER_MS = 1_000_000


def monotonic_ms() -> int:
    return time.monotonic_ns() // NS_PER_MS


def main() -> int:
    button_settings = gpiod.LineSettings(
        direction=Direction.INPUT,
        active_low=True,
    )
    led_settings = gpiod.LineSettings(
        direction=Direction.OUTPUT,
        drive=Drive.PUSH_PULL,
        active_low=True,
        output_value=Value.INACTIVE,
    )

    button_request = gpiod.request_lines(
        BUTTON_DEVICE,
        consumer="button-input",
        config={BUTTON_OFFSET: button_settings},
    )
    led_request = gpiod.request_lines(
        LED_DEVICE,
        consumer="led-output",
        config={LED_OFFSET: led_settings},
    )

    try:
        print("Press the button to toggle the outer green LED")
        last_button_value = Value.INACTIVE
        is_led_active = False
        button_state = Value.INACTIVE
        last_debounce_time = monotonic_ms()

        while True:
            button_value = button_request.get_value(BUTTON_OFFSET)
            if button_value is not last_button_value:
                last_debounce_time = monotonic_ms()

            if monotonic_ms() - last_debounce_time > DEBOUNCE_MS:
                if button_value != button_state:
                    button_state = button_value
                    if button_state is Value.ACTIVE:
                        print("Button pressed, toggling LED")
                        is_led_active = not is_led_active

            led_value = Value.ACTIVE if is_led_active else Value.INACTIVE
            led_request.set_value(LED_OFFSET, led_value)
            last_button_value = button_value
    except KeyboardInterrupt:
        print("Aborting!")
    finally:
        button_request.release()
        led_request.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
