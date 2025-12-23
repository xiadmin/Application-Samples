"""

TODO: Lorem ipsum

"""
import time
import gpiod
from gpiod.line import Direction, Drive, Value

DEVICE = "/dev/gpiochip0"
OFFSET = 106

request = gpiod.request_lines(
    DEVICE,
    consumer="simple-output",
    config={OFFSET: gpiod.LineSettings(direction=Direction.OUTPUT,
                                       drive=Drive.PUSH_PULL,
                                        active_low=True,
                                        output_value=Value.INACTIVE)},
)

try:
    print("Blinking edge LED GREEN.")
    for i in range(100):
        request.set_value(OFFSET,Value.ACTIVE)
        time.sleep(0.4)
        request.set_value(OFFSET,Value.INACTIVE)
        time.sleep(0.4)    
except KeyboardInterrupt:
    print("Aborted.")
finally:
    request.release()