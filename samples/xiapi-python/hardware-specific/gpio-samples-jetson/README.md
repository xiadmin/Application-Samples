# GPIO samples for Jetson — Python samples

GPIO, I2C, SPI, and UART examples for the XIMEA Jetson kit carrier board.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux on Nvidia Jetson |
| Hardware | XEC-NX-3P-X2G3 carrier board and supported peripherals |
| XIMEA SDK | Not required unless a script is extended to access a XIMEA camera |
| Python | 3.11+ |
| Libraries | See `requirements.txt`; `install_libi2c.sh` helps install libi2c prerequisites |

---

## Build

No build step is required for these Python scripts.

Install the documented Python dependencies before running a script:

```bash
cd samples/xiapi-python/hardware-specific/gpio-samples-jetson
python3 -m pip install -r requirements.txt
```

---

## Run

Run the script that matches the hardware interface you want to test:

```bash
cd samples/xiapi-python/hardware-specific/gpio-samples-jetson
python3 blinky.py
python3 simple_input.py
python3 polling_button.py
python3 bmp280_i2c.py
python3 bmp280_spi.py
python3 uart_loopback.py
python3 uart_p2p_jetson.py
python3 uart_p2p_pc.py
```

---

## Expected output

Each script prints status for the selected GPIO, I2C, SPI, or UART operation. Exact output depends on the connected Jetson carrier board and peripheral wiring.

---

## Known limitations / caveats

- Jetson-specific: these scripts assume Nvidia Jetson GPIO/I2C/SPI/UART interfaces and the carrier-board wiring documented in `Mapping_table.md`.
- Some scripts require connected peripherals, loopback wiring, or elevated permissions for device access.

---

## Links

- [Mapping table](Mapping_table.md)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
