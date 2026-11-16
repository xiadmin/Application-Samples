# display-video-opencv — Python sample

Shows a live MONO8 or RGB24 video in an OpenCV window with the elapsed time drawn on each frame until ESC is pressed.

| Item | Value |
|------|-------|
| Category | Display / OpenCV integration |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI Python |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Any XIMEA USB3 / PCIe camera; a color camera for `--format rgb24` |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |
| Libraries | numpy, opencv-python (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Third-party libraries

This sample uses the pip package(s) **numpy**, **opencv-python**. It never installs anything itself; without them it stops and points here. Provide them one of these ways.

**Download them, checked against pinned hashes.** `dependencies/python-constraints.txt` records the SHA-256 of every file PyPI publishes for each pinned version, and pip installs only files that match. Applying those hashes to the bare names in `requirements.txt` needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment:

```bash
cd samples/xiapi-python/cross-platform/display-video-opencv
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

Optional arguments: `--format` selects `mono8` (default) or `rgb24`; `--exposure-us` sets the exposure (default 20000).

```bash
cd samples/xiapi-python/cross-platform/display-video-opencv
python main.py
python main.py --format rgb24 --exposure-us 10000
```

---

## Expected output

The sample shows live video in a resizable window that is at most 1920x1280 pixels. The time since the start of acquisition is drawn in the top-left corner, scaled to the frame height. Press ESC in the window, close the window, or press Ctrl+C in the terminal to stop. The sample then prints the number of frames shown and the average frame rate.

```text
Opening camera index 0
Showing video. Press ESC in the window or close it to exit
Showed 412 frames in 13.8 s (29.9 fps)
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0.
- `rgb24` needs a color camera.
- The reported frame rate is the display rate. It can be lower than the camera frame rate, because drawing every frame takes time.
- `opencv-python-headless` and systems without a display cannot open the window; the sample reports an error.
- OpenCV is imported after the camera check, so a machine without a camera reports the missing camera first.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
