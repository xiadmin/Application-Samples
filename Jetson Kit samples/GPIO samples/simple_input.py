"""
Read one GPIO input line using libgpiod's Python bindings.
Button sample demonstrating simple input pin using gpiod library

Using edge button

Chip:   /dev/gpiochip
Line:   offset

TODO: Lorem ipsum

https://libgpiod.readthedocs.io/en/latest/python_api.html
"""
import time
import gpiod
from gpiod.line import Direction, Drive

DEVICE = "/dev/gpiochip1"
OFFSET = 25

request = gpiod.request_lines(
    DEVICE,
    consumer="simple-input",
    config={OFFSET: gpiod.LineSettings(direction=Direction.INPUT,drive=Drive.PUSH_PULL)},
)

try:
    for i in range(100):
        value = request.get_value(OFFSET)
        print(f"Edge button value: {value.value}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Aborted.")
finally:
    request.release()


