# capture-10-images-to-dng — Python sample

Captures 10 raw images and saves them as DNG files using the XIMEA DNG Store library.

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
| DNG Store | The "DNG Store" component of the XIMEA SP (installed by default) |

The `ximea` package comes from the XIMEA Software Package (SP) installer. `xiapi_dng.py` in this folder is a ctypes wrapper for the native DNG Store library; it needs no pip packages.

---

## Build

No build step is required for this Python sample.

---

## Run

Run from the directory where the DNG files should be created:

```bash
cd samples/xiapi-python/cross-platform/capture-10-images-to-dng
python main.py
```

---

## Expected output

The sample captures 10 RAW8 frames, Bayer data on a color camera, and saves each as `image000.dng` through `image009.dng`. Camera metadata (model, serial number, color filter array, white balance) is read once with `xidngFillMetadataFromCameraParams` and written into every file. DNG files open in raw converters such as Adobe Lightroom, darktable or RawTherapee.

```text
Opening camera index 0
Saved frame 1/10 (1280x1024) to image000.dng
...
Saved frame 10/10 (1280x1024) to image009.dng
Done
```

---

## Known limitations / caveats

- `xiapi_dng.py` is ported from the SP's `samples/_libs/xiAPI-Python/xiapi_dng.py`. The original structure omits the trailing `additionalPrivateData` fields, so the library writes 12 bytes past its end; the port includes them.
- The library is loaded on first use from `XIMEA_SP_PATH` on Windows, `/opt/XIMEA/lib` on Linux and `/Library/Frameworks/m3api.framework/Libraries` on macOS. Only Windows is verified.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Existing `image000.dng` through `image009.dng` files in the current directory are overwritten.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
