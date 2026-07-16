# capture-10-images — Python sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.32+ |
| Python | 3.11+ |

No separate pip install is needed — the `ximea` package is placed into `site-packages/ximea`
by the XIMEA SDK installer.

---

## Build

No build step is required for this Python sample.

---

## Run

### From source

```bash
python main.py
```

### From repository build checks

`scripts/build.py` runs a Python syntax check for this sample:

```bash
python3 scripts/build.py --sample xiapi-python-cross-platform-capture-10-images
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
