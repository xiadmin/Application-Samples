# capture-10-images — Python sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.32+ |
| Python | 3.9+ |

No separate pip install is needed — the `ximea` package is placed into `site-packages/ximea`
by the XIMEA SDK installer.

---

## Run

### From source

```bash
python main.py
```

### After build.ps1 (Windows)

`build.ps1` checks for the `ximea` module and generates a launcher script:

```powershell
.\build\acquisition-capture-10-images-python\run.ps1
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

-

---

## Links

- [xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
