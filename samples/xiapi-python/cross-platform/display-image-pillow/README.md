# display-image-pillow — Python sample

Captures one MONO8, RGB24 or RGB32 image, converts it to a Pillow image and opens it in the system image viewer.

| Item | Value |
|------|-------|
| Category | Display / Pillow integration |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI Python |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Any XIMEA USB3 / PCIe camera; a color camera for `--format rgb24` and `--format rgb32` |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |
| Libraries | numpy, pillow (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Third-party libraries

This sample uses the pip package(s) **numpy**, **pillow**. It never installs anything itself; without them it stops and points here. Provide them one of these ways.

**Download them, checked against pinned hashes.** `dependencies/python-constraints.txt` records the SHA-256 of every file PyPI publishes for each pinned version, and pip installs only files that match. Applying those hashes to the bare names in `requirements.txt` needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment:

```bash
cd samples/xiapi-python/cross-platform/display-image-pillow
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

Optional argument: `--format` selects `mono8` (default), `rgb24` or `rgb32`.

```bash
cd samples/xiapi-python/cross-platform/display-image-pillow
python main.py
python main.py --format rgb24
```

---

## Expected output

The sample captures one image with a 10 ms exposure and opens it with `PIL.Image.show()`, which writes a temporary file and opens it in the default image viewer. MONO8 becomes a Pillow `L` image; RGB24 and RGB32 become `RGB` images. xiAPI delivers color data as BGR(A), so the sample reads it with `invert_rgb_order=True`. For color formats, automatic white balance is enabled and 10 frames are discarded first so it can settle.

```text
Opening camera index 0
Captured mono8 image 1280x1024
Opening the L image in the system image viewer
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0.
- `rgb24` and `rgb32` need a color camera.
- The fourth byte of RGB32 is not an opacity value, so the sample drops it and shows an `RGB` image.
- Pillow has no 16-bit RGB modes, so the sample offers only 8-bit formats.
- Without a configured default image viewer, `show()` may do nothing.
- Pillow is imported after the camera check, so a machine without a camera reports the missing camera first.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
