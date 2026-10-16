# capture-from-two-cameras — Python sample

Opens the first two connected cameras, splits the interface bandwidth between them, captures one image from each and prints its size and first pixels.

| Item | Value |
|------|-------|
| Category | Multi-camera / basic acquisition |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI Python |
| Language | Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Two XIMEA USB3 / PCIe cameras |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Libraries | None |
| Python | 3.11+ |

The sample needs only the `ximea` package, which comes from the XIMEA Software Package (SP) installer. It is not a pip package.

---

## Build

No build step is required for this Python sample.

---

## Run

Optional argument: `--exposure-us` sets the exposure of both cameras (default 10000).

```bash
cd samples/xiapi-python/cross-platform/capture-from-two-cameras
python main.py
python main.py --exposure-us 20000
```

---

## Expected output

The sample opens camera indexes 0 and 1. Both cameras usually share one USB or PCIe controller, so it reads the bandwidth limit of the first camera, gives each camera half of it, and enables the limit. Both cameras capture one MONO8 image.

```text
Opening cameras...
Camera 1: serial number 12345678, bandwidth limit 1500 Mbit/s, exposure 10000 us
Camera 2: serial number 23456789, bandwidth limit 1500 Mbit/s, exposure 10000 us
Starting data acquisition...
Camera 1: image 1280x1024 received
Camera 1: first 10 pixels: [12, 13, 12, 14, 13, 12, 12, 13, 14, 12]
Camera 2: image 1280x1024 received
Camera 2: first 10 pixels: [11, 12, 12, 11, 13, 12, 11, 12, 13, 12]
Done
```

Serial numbers, resolutions and pixel values depend on the cameras.

---

## Known limitations / caveats

- At least two cameras must be connected; the sample exits with an error otherwise. Extra cameras are ignored.
- The bandwidth split assumes both cameras share one controller. On separate controllers each camera could use its full bandwidth, so the split may lower the reachable frame rate.
- Both cameras are always released, even when a call fails on one of them.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
