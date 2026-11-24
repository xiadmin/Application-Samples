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

This sample uses the pip package(s) **numpy**. Nothing is downloaded unless you ask for it; without them the sample stops and explains both ways.

**Install them yourself**, any way you like; the sample only imports them. Copies you provide are not checked, and their versions may differ from the tested ones. For example, from your own wheel files in `dependencies/local/wheels/` at the repository root, without network access:

```bash
cd samples/xiapi-python/cross-platform/capture-10-images
python -m pip install --no-index --find-links ../../../../dependencies/local/wheels numpy
```

Or install them with your system package manager.

To install the tested versions instead, checked against the SHA-256 of every file PyPI publishes for them (recorded in `dependencies/python-constraints.txt`), use `requirements.txt`. Applying those hashes to its bare names needs pip 26.3 or newer, so update pip first, preferably inside a virtual environment. `requirements.txt` always applies the hashes, also to wheels in `dependencies/local/wheels/`.

```bash
python -m pip install --upgrade "pip>=26.3"
python -m pip install --require-hashes -r requirements.txt
```

**Or let the sample install them.** With `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` set, a sample that misses a package runs that same `pip install --require-hashes -r requirements.txt` itself, with the Python that runs the sample, and continues. A hash mismatch, a pip older than 26.3 or an environment pip may not install into stops it with pip's message.

To let this and every other sample download what it needs, set `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` to `1` (`ON`, `TRUE` and `YES` work too) before you build or run; the commands below then work unchanged. Unset it to go back to your own copies.

```bash
export AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1   # Linux/macOS
```

```powershell
$env:AUTOMATICALLY_DOWNLOAD_DEPENDENCIES = '1'   # Windows PowerShell
```

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
