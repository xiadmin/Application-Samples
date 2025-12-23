"""
TODO: Lorem ipsum

This would need interrupts. They need to be configured in the Tegra image.
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
        edge_detection=Edge.BOTH,
        debounce_period=timedelta(milliseconds=DEBOUNCE_MS),
        bias=Bias.PULL_DOWN
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
        config={
            BUTTON_OFFSET: button_settings
        }
    )

led_request = gpiod.request_lines(
        LED_DEVICE,
        consumer="LED-output",
        config={LED_OFFSET: led_settings},
)

try:
    print("Press the button to activate EDGE LED")

    while True:
        ret_value = button_request.wait_edge_events(3.0)

        if ret_value:
            print("Event ready")
        else:
            print("no events")
            continue 
        
        event = button_request.read_edge_events()

        if event.event_type is gpiod.EdgeEvent.Type.RISING_EDGE:
            led_request.set_value(LED_OFFSET, Value.INACTIVE) 
        elif event.event_type is gpiod.EdgeEvent.Type.FALLING_EDGE:
            led_request.set_value(LED_OFFSET, Value.ACTIVE)         

except KeyboardInterrupt:
    print("Aborted.")
finally:
    button_request.release()
    led_request.release()

