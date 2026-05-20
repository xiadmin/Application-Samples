"""
This is a sample code demonstrating the use of a button on the Jetson Kit. 
The button press detection is based on polling with debouncing.
Should an interrupt be needed, a reconfiguration of the Tegra image is required. 
Please contact XIMEA support for more details.

Outer button (GPIO10) and Outer green LED (GPIO11) are used in the sample.
Libgpiod is used for GPIO control.

Workflow:
- Initialize GPIO10 for input, inverse logic
- Initialize GPIO11 for output, inverse logic
- In a loop:
-- On a proper button press (with debouncing), toggle the LED state

Abort the program with Ctrl+C.

Sample is based on: 
https://docs.arduino.cc/built-in-examples/digital/Debounce/

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import time
import gpiod
from gpiod.line import Direction, Drive, Value

# Using outer button, GPIO10/PEE.02 on Orin NX, which is mapped to gpiochip1, line offset 25
BUTTON_DEVICE = "/dev/gpiochip1"
BUTTON_OFFSET = 25

# Using outer green LED, GPIO11/PQ.06 on Orin NX, which is mapped to gpiochip0, line offset 106 
LED_DEVICE = "/dev/gpiochip0"
LED_OFFSET = 106

DEBOUNCE_MS = 50

button_settings = gpiod.LineSettings(
        direction=Direction.INPUT,
        active_low=True # Buttons use inverse logic
    )

led_settings = gpiod.LineSettings(
        direction=Direction.OUTPUT,
        drive=Drive.PUSH_PULL,
        active_low=True, # LEDs use inverse logic
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
    print("Press the button to activate outer green LED")

    last_button_value = Value.INACTIVE
    led_state = False
    button_state = Value.INACTIVE
    last_debounce_time = time.monotonic_ns() // 1_000_000 

    while True:
    	# Reading button value
        button_value = button_request.get_value(BUTTON_OFFSET)
        
        if button_value is not last_button_value:
            last_debounce_time = time.monotonic_ns() // 1_000_000 
        
        current_time = time.monotonic_ns() // 1_000_000  
        
        if (current_time - last_debounce_time) > DEBOUNCE_MS:
            if button_value != button_state:
                button_state = button_value
                if button_state is Value.ACTIVE:
                    # Button pressed, toggle LED state
                    print("Button pressed, toggling LED")
                    led_state = not led_state

        # Setting LED state
        if led_state==True:
            led_request.set_value(LED_OFFSET, Value.ACTIVE)
        else:
            led_request.set_value(LED_OFFSET, Value.INACTIVE)

        last_button_value = button_value           

except KeyboardInterrupt:
    print("Aborting!")

finally:
    button_request.release()
    led_request.release()

