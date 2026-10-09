# display-image-opencv — Python sample

Captures one MONO8, RGB24 or MONO16 image and shows it in an OpenCV window scaled to fit the screen.

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
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |
| Libraries | numpy, opencv-python (versions pinned in `dependencies/python-constraints.txt`) |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Build

Install the third-party packages. `requirements.txt` activates the central constraints file, so the versions come from `dependencies/python-constraints.txt`:

```bash
cd samples/xiapi-python/cross-platform/display-image-opencv
python -m pip install -r requirements.txt
```

---

## Run

Optional arguments: `--format` selects `mono8` (default), `rgb24` or `mono16`; `--exposure-us` sets the exposure (default 10000).

```bash
cd samples/xiapi-python/cross-platform/display-image-opencv
python main.py
python main.py --format rgb24
python main.py --format mono16 --exposure-us 20000
```

---

## Expected output

The sample captures one image, releases the camera, and shows the image in a resizable window that is at most 1920x1280 pixels. Press any key in the window or close it to exit. With `rgb24`, automatic white balance is enabled and 10 frames are discarded first so it can settle. With `mono16`, the captured range is stretched to 8 bits for display.

```text
Opening camera index 0
Captured mono8 image 1280x1024
Press any key in the window or close it to exit
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0.
- `rgb24` needs a color camera.
- The `mono16` stretch maps the darkest and brightest pixels of the frame to black and white, so the displayed brightness is not the absolute sensor value.
- `opencv-python-headless` and systems without a display cannot open the window; the sample reports an error.
- OpenCV is imported after the camera check, so a machine without a camera reports the missing camera first.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
