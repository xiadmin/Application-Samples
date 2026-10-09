# save-image-pillow — Python sample

Captures one image (RGB24 on color cameras, MONO8 on monochrome cameras) and saves it with Pillow, which picks the file format from the extension.

| Item | Value |
|------|-------|
| Category | Image saving / Pillow integration |
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
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |
| Libraries | numpy, pillow (versions pinned in `dependencies/python-constraints.txt`) |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Build

Install the third-party packages. `requirements.txt` activates the central constraints file, so the versions come from `dependencies/python-constraints.txt`:

```bash
cd samples/xiapi-python/cross-platform/save-image-pillow
python -m pip install -r requirements.txt
```

---

## Run

Optional argument: `--output` sets the output file (default `image.bmp`). The extension selects the format, for example `.bmp`, `.png`, `.jpg` or `.tif`.

```bash
cd samples/xiapi-python/cross-platform/save-image-pillow
python main.py
python main.py --output image.png
```

---

## Expected output

The sample captures one image with a 10 ms exposure and saves it to the output file. Color cameras use RGB24 with automatic white balance and discard 10 frames first so it can settle; monochrome cameras use MONO8.

```text
Opening camera index 0
Captured image 1280x1024
Saved image to image.bmp
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0.
- An existing output file is overwritten. The output directory must already exist.
- An extension Pillow does not know is reported as an error after the capture.
- Pillow is imported after the camera check, so a machine without a camera reports the missing camera first.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
