# capture-10-images — Python sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| API type | xiAPI Python |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SP | [Approved version in DEPENDENCIES.md](../../../../DEPENDENCIES.md#2-approved-dependency-table) |
| Python | 3.11+ |

The `ximea` package comes from the XIMEA Software Package (SP) installer, placed into `site-packages/ximea`. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Build

No build step is required for this Python sample.

---

## Run

```bash
python main.py
```

---

## Expected output

```
Found 1 camera(s), opening index 0
Exposure: 100000 us (100 ms)
Capturing 10 frames
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- Requires a connected XIMEA camera and installed SDK runtime; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0 and uses a fixed 100 ms exposure.

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
