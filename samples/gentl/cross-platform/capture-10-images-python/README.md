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
| Libraries | harvesters, genicam, numpy (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |
| Python | 3.11+ |

`genicam` is published as binary wheels only, and there is none for macOS on Intel, so the sample does not run there.

---

## Third-party libraries

This sample uses the pip package(s) **harvesters**, **genicam**, **numpy**. Nothing is downloaded unless you ask for it; without them the sample stops and explains both ways.

**Install them yourself**, any way you like; the sample only imports them. Copies you provide are not checked, and their versions may differ from the tested ones. For example, from your own wheel files in `dependencies/local/wheels/` at the repository root, without network access:

```bash
cd samples/gentl/cross-platform/capture-10-images-python
python -m pip install --no-index --find-links ../../../../dependencies/local/wheels harvesters genicam numpy
```

Or install them with your system package manager.

To install the tested versions instead, checked against the SHA-256 of every file PyPI publishes for them (recorded in `dependencies/python-constraints.txt`), use `requirements.txt`. Applying those hashes to its bare names needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment. `requirements.txt` always applies the hashes, also to wheels in `dependencies/local/wheels/`.

```bash
python -m pip install --upgrade "pip>=26.3"
python -m pip install --require-hashes -r requirements.txt
```

**Or let the sample install them.** With `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` set, a sample that misses a package runs that same `pip install --require-hashes -r requirements.txt` itself, with the Python that runs the sample, and continues. A hash mismatch, a pip older than 26.3 or an environment pip may not install into stops it with pip's message.

To let this and every other sample download what it needs, set `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` to `1` (`ON`, `TRUE` and `YES` work too) before you build or run; the commands below then work unchanged. Unset it to go back to your own copies.

```bash
export AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1   # Linux/macOS
```

```powershell
$env:AUTOMATICALLY_DOWNLOAD_DEPENDENCIES = '1'   # Windows PowerShell
```

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
cd samples\gentl\cross-platform\capture-10-images-python
python main.py --cti "C:\XIMEA\GenTL Producer\ximea.gentl2.cti"
```

On Linux and macOS, pass the path of the `.cti` file the XIMEA Software Package installed: `python main.py --cti <path-to-ximea.gentl2.cti>`.

| Option | Default | Meaning |
|--------|---------|---------|
| `--cti <file>` | `ximea.gentl2.cti`, else `ximea.gentl.cti`, from `GENICAM_GENTL64_PATH` | GenTL producer to load |
| `--exposure-us <us>` | 10000 | Exposure time in microseconds, written to the `ExposureTime` feature |

---

## Expected output

The sample prints the producer and device it opens, then the size, timestamp and first pixels of each frame:

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
- The sample loads one XIMEA GenTL producer; it does not iterate over every producer in the GenTL search path.

---

## Links

- [XIMEA GenTL producer](https://www.ximea.com/support/wiki/apis/GenTL)
- [harvesters documentation](https://github.com/genicam/harvesters)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
