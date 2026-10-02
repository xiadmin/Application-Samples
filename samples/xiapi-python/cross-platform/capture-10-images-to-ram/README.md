# capture-10-images-to-ram — Python sample

Captures 10 images into application-owned RAM.

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
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |

The `ximea` package comes from the XIMEA Software Package (SP) installer. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Build

No build step is required for this Python sample.

---

## Run

```bash
cd samples/xiapi-python/cross-platform/capture-10-images-to-ram
python main.py
```

---

## Expected output

The sample captures 10 RAW8 frames, copies each one into a `bytes` object, closes the camera, and then prints per-frame metadata from RAM.

```text
Opening camera index 0
Camera closed; reading retained images from RAM
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- `Image.bp` points into a driver buffer that the next `get_image()` reuses and `close_device()` frees; `get_image_data_raw()` copies it, including line padding.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
