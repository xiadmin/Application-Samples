# capture-10-images-python — Python sample

Loads the XIMEA GenTL producer with the harvesters GenTL consumer, sets the exposure and pixel format through the GenICam node map, captures 10 images and prints the size, timestamp and first pixels of each one.

| Item | Value |
|------|-------|
| Category | GenTL / Python consumer |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | GenTL |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS on Apple silicon |
| Hardware | Any XIMEA camera supported by the GenTL producer |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads), which installs the GenTL producer and sets `GENICAM_GENTL64_PATH` |
| Python | 3.11+ |
| Libraries | harvesters, genicam, numpy (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |

`genicam` is published as binary wheels only, and there is none for macOS on Intel, so the sample does not run there.

---

## Third-party libraries

This sample uses the pip package(s) **harvesters**, **genicam**, **numpy**. It never installs anything itself; without them it stops and points here. Provide them one of these ways.

**Download them, checked against pinned hashes.** `dependencies/python-constraints.txt` records the SHA-256 of every file PyPI publishes for each pinned version, and pip installs only files that match. Applying those hashes to the bare names in `requirements.txt` needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment:

```bash
cd samples/gentl/cross-platform/capture-10-images-python
python -m pip install --upgrade "pip>=26.3"
python -m pip install --require-hashes -r requirements.txt
```

**Provide them yourself.** Put the wheel files into `.deps/wheels/` at the repository root and install from there without network access; the same hash check applies:

```bash
python -m pip install --no-index --require-hashes -r requirements.txt
```

Or install the packages with your system package manager and skip `requirements.txt`; the versions then come from your distribution and may differ from the tested ones.

---

## Build

No build step is required. Install the third-party packages first; see [Third-party libraries](#third-party-libraries).

---

## Run

```bash
cd samples/gentl/cross-platform/capture-10-images-python
python main.py
python main.py --exposure-us 20000
```

To load a specific producer, pass its path with `--cti`. On Windows, for example:

```powershell
cd samples/gentl/cross-platform/capture-10-images-python
python main.py --cti "C:\XIMEA\GenTL Producer\ximea.gentl2.cti"
```

On Linux and macOS, pass the path of the `.cti` file the XIMEA Software Package installed: `python main.py --cti <path-to-ximea.gentl2.cti>`.

| Option | Default | Meaning |
|--------|---------|---------|
| `--cti <file>` | `ximea.gentl2.cti`, else `ximea.gentl.cti`, from `GENICAM_GENTL64_PATH` | GenTL producer to load |
| `--exposure-us <us>` | 10000 | Exposure time in microseconds, written to the `ExposureTime` feature |

---

## Expected output

```text
Loading GenTL producer C:\XIMEA\GenTL Producer\ximea.gentl2.cti
Opening device MC031CG-SY-UB
Pixel format Mono8, exposure 10000 us
Frame 1/10: 2064x1544, timestamp 18446744, first pixels [21, 23, 22, 20, 24, 22, 21, 23]
...
Done
```

---

## Known limitations / caveats

- The producer lookup matches the C++ GenTL samples: `--cti`, otherwise the directories in `GENICAM_GENTL64_PATH` (`GENICAM_GENTL32_PATH` for 32-bit Python), preferring the current producer over the legacy one.
- The sample opens the first device the producer reports.
- `PixelFormat` is set to `Mono8` when the camera offers it; otherwise the camera's default format is kept and the printed pixel values are that format's raw samples.
- The timestamp is printed in device ticks, as the producer reports it.
- harvesters is imported after the producer lookup, so a machine without the XIMEA Software Package reports the missing producer first.
- Ported from the internal `simple_acq_sample`, which looped over every producer in the search path; this version loads one XIMEA producer.

---

## Links

- [XIMEA GenTL producer](https://www.ximea.com/support/wiki/apis/GenTL)
- [harvesters documentation](https://github.com/genicam/harvesters)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
