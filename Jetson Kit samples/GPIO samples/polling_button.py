"""
TODO: Lorem ipsum

This would need interrupts. They need to be configured in the Tegra image.
based on 
https://docs.arduino.cc/built-in-examples/digital/Debounce/

Reference:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import time
from datetime import timedelta
import gpiod
from gpiod.line import Direction, Drive, Edge, Value, Bias


BUTTON_DEVICE = "/dev/gpiochip1"
BUTTON_OFFSET = 25

LED_DEVICE = "/dev/gpiochip0"
LED_OFFSET = 106

DEBOUNCE_MS = 50


button_settings = gpiod.LineSettings(
        direction=Direction.INPUT,
        active_low=True # inverse logic for buttons
    )

led_settings = gpiod.LineSettings(
        direction=Direction.OUTPUT,
        drive=Drive.PUSH_PULL,
        active_low=True,
        output_value=Value.INACTIVE
    )

button_request = gpiod.request_lines(
        BUTTON_DEVICE,
        consumer="button-input",
        config={BUTTON_OFFSET: button_settings}
    )

led_request = gpiod.request_lines(
        LED_DEVICE,
        consumer="LED-output",
        config={LED_OFFSET: led_settings},
)

try:
    print("Press the button to activate EDGE LED")

    last_button_value = Value.INACTIVE
    led_state = False
    button_state = Value.INACTIVE
    last_debounce_time = time.monotonic_ns() // 1_000_000 

    while True:
    	# Replace this to polling
        button_value = button_request.get_value(BUTTON_OFFSET)
        
        if button_value is not last_button_value:
            last_debounce_time = time.monotonic_ns() // 1_000_000  # Convert to milliseconds
        
        current_time = time.monotonic_ns() // 1_000_000  # Convert to milliseconds
        
        if (current_time - last_debounce_time) > DEBOUNCE_MS:
            if button_value != button_state:
                button_state = button_value
                if button_state is Value.ACTIVE:
                    # Button pressed, toggle LED state
                    print("Button pressed, toggling LED")
                    led_state = not led_state

        if led_state==True:
            led_request.set_value(LED_OFFSET, Value.ACTIVE)
        else:
            led_request.set_value(LED_OFFSET, Value.INACTIVE)

        last_button_value = button_value           

except KeyboardInterrupt:
    print("Aborted.")
finally:
    button_request.release()
    led_request.release()

