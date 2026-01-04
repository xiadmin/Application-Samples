"""
This is a sample code demonstrating the use of input GPIO on the Jetson Kit.
Outer button (GPIO10) is used in the sample, to avoid the need for any hardware setup.
Libgpiod is used for GPIO control.

Workflow:
- Initialize GPIO line for input, inverse logic
- Read the button state in a loop

Abort the program with Ctrl+C.

Libgpiod documentation:
https://libgpiod.readthedocs.io/en/latest/python_api.html
"""
import time
import gpiod
from gpiod.line import Direction, Drive

#Using outer button, GPIO10/PEE.02 on Orin NX, which is mapped to gpiochip1, line offset 25
DEVICE = "/dev/gpiochip1"
OFFSET = 25

request = gpiod.request_lines(
    DEVICE,
    consumer="simple-input",
    config={OFFSET: gpiod.LineSettings(direction=Direction.INPUT, 
                                       active_low=True # Buttons use inverse logic
                                       )},
    )

try:
    for i in range(100):
        # Reading value
        value = request.get_value(OFFSET)
        print(f"Outer button value: {value.value}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Aborting!")

finally:
    request.release()


