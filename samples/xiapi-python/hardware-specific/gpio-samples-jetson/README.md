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
| Libraries | gpiod, spidev, pyserial (versions pinned in `dependencies/python-constraints.txt`); pylibi2c, built from source by `install_libi2c.sh` |

---

## Build

No build step is required for these Python scripts.

Install the documented Python dependencies before running a script:

```bash
cd samples/xiapi-python/hardware-specific/gpio-samples-jetson
python3 -m pip install -r requirements.txt
```

`requirements.txt` references `dependencies/python-constraints.txt` at the repository root via a `-c` directive. That file pins the approved package versions (`gpiod`, `spidev`, `pyserial`); update it there to change versions across all samples.

The same file also points at the shared wheelhouse, `.deps/wheels/` at the repository root, which pip ignores while it is empty. To install without network access, fill it on the Jetson itself once — wheels are specific to platform and Python version:

```bash
python3 -m pip download -r requirements.txt -d ../../../../.deps/wheels   # once, online
python3 -m pip install --no-index -r requirements.txt                     # offline
```

`bmp280_i2c.py` imports `pylibi2c`, which is not in `requirements.txt`. `install_libi2c.sh` clones [libi2c](https://github.com/amaork/libi2c) into `~/Downloads`, checks out the pinned commit `dcdf6c6` and installs it with `sudo python setup.py install`. It needs `git`, network access, sudo rights and an existing `~/Downloads` directory:

```bash
cd samples/xiapi-python/hardware-specific/gpio-samples-jetson
bash install_libi2c.sh
```

---

## Run

Run the script that matches the hardware interface you want to test. Each script needs one package:

| Script | Interface | Package |
|--------|-----------|---------|
| `blinky.py`, `simple_input.py`, `polling_button.py` | GPIO | gpiod |
| `bmp280_spi.py` | SPI | spidev |
| `uart_loopback.py`, `uart_p2p_jetson.py`, `uart_p2p_pc.py` | UART | pyserial |
| `bmp280_i2c.py` | I2C | pylibi2c |

`uart_p2p_pc.py` is the PC side of the PING/PONG pair: run it on the PC connected to the Jetson UART0, not on the Jetson, while `uart_p2p_jetson.py` runs on the Jetson. The PC needs pyserial too (`python -m pip install -r requirements.txt` from this folder installs it there), and `PORT` at the top of the script (default `/dev/ttyUSB0`) must name the PC's serial adapter.

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
