# capture-10-images — Python sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

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
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Python | 3.11+ |
| Libraries | numpy (versions and SHA-256 hashes pinned in `dependencies/python-constraints.txt`) |

The `ximea` package comes from the XIMEA Software Package (SP) installer, placed into `site-packages/ximea`. It is not a pip package and does not appear in `dependencies/python-constraints.txt`.

---

## Third-party libraries

This sample uses the pip package(s) **numpy**. It never installs anything itself; without them it stops and points here. Provide them one of these ways.

**Download them, checked against pinned hashes.** `dependencies/python-constraints.txt` records the SHA-256 of every file PyPI publishes for each pinned version, and pip installs only files that match. Applying those hashes to the bare names in `requirements.txt` needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment:

```bash
cd samples/xiapi-python/cross-platform/capture-10-images
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

```bash
cd samples/xiapi-python/cross-platform/capture-10-images
python main.py
```

---

## Expected output

```text
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
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
