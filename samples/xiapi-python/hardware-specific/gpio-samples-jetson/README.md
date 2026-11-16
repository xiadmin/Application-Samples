# GPIO samples for Jetson — Python samples

GPIO, I2C, SPI, and UART examples for the XIMEA Jetson kit carrier board.

| Item | Value |
|------|-------|
| Category | Trigger / GPIO / I/O / strobe |
| OS platform | Linux (NVIDIA JetPack) |
| Hardware platform | NVIDIA Jetson |
| API type | None (Jetson GPIO/I2C/SPI/UART; no xiAPI calls) |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux on Nvidia Jetson |
| Hardware | XEC-NX-3P-X2G3 carrier board and supported peripherals |
| XIMEA SDK | Not required unless a script is extended to access a XIMEA camera |
| Python | 3.11+ |
| Libraries | gpiod, smbus2, spidev, pyserial (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |

---

## Third-party libraries

This sample uses the pip package(s) **gpiod**, **smbus2**, **spidev**, **pyserial**. It never installs anything itself; without them it stops and points here. Provide them one of these ways.

**Download them, checked against pinned hashes.** `dependencies/python-constraints.txt` records the SHA-256 of every file PyPI publishes for each pinned version, and pip installs only files that match. Applying those hashes to the bare names in `requirements.txt` needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment:

```bash
cd samples/xiapi-python/hardware-specific/gpio-samples-jetson
python3 -m pip install --upgrade "pip>=26.3"
python3 -m pip install --require-hashes -r requirements.txt
```

**Provide them yourself.** Put the wheel files into `.deps/wheels/` at the repository root and install from there without network access; the same hash check applies:

```bash
python3 -m pip install --no-index --require-hashes -r requirements.txt
```

Or install the packages with your system package manager and skip `requirements.txt`; the versions then come from your distribution and may differ from the tested ones.

---

## Build

No build step is required for these Python scripts.

Install the documented Python dependencies before running a script:

```bash
cd samples/xiapi-python/hardware-specific/gpio-samples-jetson
python3 -m pip install --require-hashes -r requirements.txt
```

`requirements.txt` references `dependencies/python-constraints.txt` at the repository root via a `-c` directive. That file pins the approved package versions and the SHA-256 hashes of their files (`gpiod`, `smbus2`, `spidev`, `pyserial`); update it there to change versions across all samples.

The same file also points at the shared wheelhouse, `.deps/wheels/` at the repository root, which pip ignores while it is empty. To install without network access, fill it on the Jetson itself once — wheels are specific to platform and Python version:

```bash
python3 -m pip download -r requirements.txt -d ../../../../.deps/wheels       # once, online
python3 -m pip install --no-index --require-hashes -r requirements.txt   # offline
```

---

## Run

Run the script that matches the hardware interface you want to test. Each script needs one package:

| Script | Interface | Package |
|--------|-----------|---------|
| `blinky.py`, `simple_input.py`, `polling_button.py` | GPIO | gpiod |
| `bmp280_spi.py` | SPI | spidev |
| `uart_loopback.py`, `uart_p2p_jetson.py`, `uart_p2p_pc.py` | UART | pyserial |
| `bmp280_i2c.py` | I2C | smbus2 |

`uart_p2p_pc.py` is the PC side of the PING/PONG pair: run it on the PC connected to the Jetson UART0, not on the Jetson, while `uart_p2p_jetson.py` runs on the Jetson. The PC needs pyserial too (`python -m pip install --require-hashes -r requirements.txt` from this folder installs it there), and `PORT` at the top of the script (default `/dev/ttyUSB0`) must name the PC's serial adapter.

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

- Jetson-specific: these scripts assume Nvidia Jetson GPIO/I2C/SPI/UART interfaces and the carrier-board wiring. The device paths, line offsets, bus numbers and serial ports are constants at the top of each script (for example `DEVICE`/`OFFSET`, `I2C_BUS`, `BUS`/`CHIP_SELECT`, `PORT`), set for Orin NX; check them against the carrier board pin mapping in the XIMEA documentation before use.
- Some scripts require connected peripherals, loopback wiring, or elevated permissions for device access.

---

## Links

- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
