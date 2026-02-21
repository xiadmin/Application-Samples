# GPIO mapping table

There are four parts of Jetson-kit that contain GPIO pins.
- GPIO header
- Dip switches
- Buttons
- LEDs

## GPIO Pin mappings

Note that the GPIO pins use 1V8 logic.

### GPIO header:

| *GPIO header pin number* |  *GPIO name* | *SOC Pin number* |  *GPIO label* | *GPIO chip/device name* | *GPIO offset* |
| --- | --- | --- | --- | --- |  --- |  
| 11    | GPIO3 | 126 | PCC.00  | gpiochip1 | 12 |
| 12    | GPIO4 | 127 | PCC.01  | gpiochip1 | 13 |
| 13    | GPIO5 | 128 | PCC.02  | gpiochip1 | 14 |
| 14    | GPIO6 | 130 | PCC.03  | gpiochip1 | 15 |
| 15    | GPIO7 | 206 | PG.06   | gpiochip0 | 41 |

### Dip switches:

| *GPIO header pin number* |  *GPIO name* | *SOC Pin number* |  *GPIO label* | *GPIO chip/device name* | *GPIO offset* |
| --- | --- | --- | --- | --- |  --- |  
| 12    | GPIO13 | 228 | PH.00  | gpiochip0 | 43 |

### Buttons:

Note that buttons use inverted logic - 0 means not pressed, 1 means pressed.

| *Description* |  *GPIO name* | *SOC Pin number* |  *GPIO label* | *GPIO chip/device name* | *GPIO offset* |
| --- | --- | --- | --- | --- |  --- |
| Outer button    | GPIO10 | 212 | PEE.02  | gpiochip1 | 25 |
| Inner button    | GPIO12 | 218 | PN.01  | gpiochip0 | 85 |

Please do note that the pins are not specifically configured to detect interrupts. Should this feature be needed on specific pins, the Tegra image needs to be modified. Contact XIMEA support for assistance.

### LEDs:

| *Description* |  *GPIO name* | *SOC Pin number* |  *GPIO label* | *GPIO chip/device name* | *GPIO offset* |
| --- | --- | --- | --- | --- |  --- |  
| Outer LED red    | GPIO9 | 211 | PAC.06  | gpiochip0 | 144 |
| Outer LED green    | GPIO11 | 216 | PQ.06 | gpiochip0 | 106 |
| Inner LED red    | GPIO2 | 124 | PP.06 | gpiochip0 | 98 |
| Inner LED green    | GPIO1 | 118 | PQ.05  | gpiochip0 | 105 |

## UART

There are three UART units total:
* Debug UART
* UART0 with RTS# and CTS# pins
* UART1 

Note that the UART uses 1V8 logic.

### Debug UART

In /dev/ debug UART is *ttyTCU0* 

| *Parameter* | *Value* |
| --- | --- |
| Baud rate | 115200 |
| Data bits | 8 |
| Stop bits | 1 |
| Parity | None |

Debug UART usage is further documented in the technical manual.

### UART0

In ls /dev/ the UART0 is device *ttyTHS3*

| *GPIO header pin number* | *Function* |
| --- | --- |
| 16 | RTS# |
| 17 | CTS# |
| 18 | RX |
| 19 | TX |

Note:
The mapping to ttyTHS3 is available from Tegra image 2026-02-10.
Tegra image 2025-12-17 has UART0 mapped to ttyS3.

### UART1

In ls /dev/ the UART0 is device *ttyTHS1*

| *GPIO header pin number* | *Function* |
| --- | --- |
| 20 | TX |
| 21 | RX |

## I2C

The Jetson kit has two I2C buses:
* I2C1
* I2C2

Note that I2C1 uses 3V3 logic, while I2C2 uses 1V8 logic.

### I2C1

In ls /dev/ the I2C1 is device *i2c-7*

| *GPIO header pin number* | *Function* |
| --- | --- |
| 27 | SCL |
| 28 | SDA |


### I2C2

In ls /dev/ the I2C2 is device *i2c-0*

| *GPIO header pin number* | *Function* |
| --- | --- |
| 29 | SCL |
| 30 | SDA |

## SPI

Jetson-kit has a single SPI bus routed to the GPIO header. The SPI has settable logic level - 1V8 or 3V3, switchable with DIP switch.

In /dev/ the SPI is device *spidev0.0* (Chip select 0) and *spidev0.1* (Chip select 1)

| *GPIO header pin number* | *Function* |
| --- | --- |
| 22 | MISO |
| 23 | CLK |
| 24 | MOSI |
| 25 | CS0 |
| 26 | CS1 |

Currectly ovly 3V3 logic has been tested.

## Notes:

Please note that this mapping table is valid from Tegra image ver 2025-12-17 and beyond. Should your image be older, please contact the XIMEA support for an update.

For the samples the work without sudo, the user needs to be added to the following groups:
- gpio
- i2c
- dialout

The user can be added to a group with:
_sudo usermod -aG group nvidia_

The user is added automatically the these groups from SSD image version 2026-01-06.

