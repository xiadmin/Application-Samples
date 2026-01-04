"""
This is a sample code demonstrating the use of output GPIO on the Jetson Kit.
Outer green LED (GPIO11) is used in the sample, to avoid the need for any hardware setup.
Libgpiod is used for GPIO control.

Workflow:
- Initialize GPIO line for output, inverse logic
- Blink the LED in a loop

Abort the program with Ctrl+C.

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""

import time
import gpiod
from gpiod.line import Direction, Drive, Value

#Using outer green LED, GPIO11/PQ.06 on Orin NX, which is mapped to gpiochip0, line offset 106 
DEVICE = "/dev/gpiochip0"
OFFSET = 106

request = gpiod.request_lines(
    DEVICE,
    consumer="simple-output",
    config={OFFSET: gpiod.LineSettings(direction=Direction.OUTPUT,
                                       drive=Drive.PUSH_PULL,
                                        active_low=True, #LEDs use inverse logic
                                        output_value=Value.INACTIVE)},
)

try:
    print("Blinking outer LED GREEN.")

    for i in range(100):
        request.set_value(OFFSET,Value.ACTIVE)
        time.sleep(0.4)
        request.set_value(OFFSET,Value.INACTIVE)
        time.sleep(0.4)

except KeyboardInterrupt:
    print("Aborting!")

finally:
    request.release()