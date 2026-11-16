# capture-10-images-to-file — Python sample

Captures 10 images and saves them as TIFF files using tifffile.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI Python |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |
| Libraries | numpy, tifffile (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Third-party libraries

This sample uses the pip package(s) **numpy**, **tifffile**. It never installs anything itself; without them it stops and points here. Provide them one of these ways.

**Download them, checked against pinned hashes.** `dependencies/python-constraints.txt` records the SHA-256 of every file PyPI publishes for each pinned version, and pip installs only files that match. Applying those hashes to the bare names in `requirements.txt` needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment:

```bash
cd samples/xiapi-python/cross-platform/capture-10-images-to-file
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

Run from the directory where the TIFF files should be created:

```bash
cd samples/xiapi-python/cross-platform/capture-10-images-to-file
python main.py
```

---

## Expected output

The sample enables automatic exposure/gain, discards 10 warm-up frames, and saves the next 10 frames as `image000.tif` through `image009.tif`. Color cameras use RGB24 output with automatic white balance; monochrome cameras use MONO8.

```text
Opening camera index 0
Saved frame 1/10 to image000.tif
...
Saved frame 10/10 to image009.tif
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- `tifffile` is imported after the camera check, so a machine without a camera reports the missing camera first.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
