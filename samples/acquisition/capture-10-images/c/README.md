# capture-10-images — C sample

Captures 10 frames from the first available XIMEA camera at 100 ms exposure
and prints per-frame metadata to stdout.

**One-line purpose:** demonstrate basic XIMEA xiAPI acquisition in pure C (C11, single source file).

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.20 or newer (tested with 4.33). Install the XIMEA Software Package and set `XIMEA_SP_PATH` to its root directory (`C:\XIMEA` on Windows, `/opt/XIMEA` on Linux). This repository does not vendor XIMEA runtime libraries or DLLs. |
| CMake | 3.16 or newer |
| Compiler | MSVC 2019+, GCC 9+, or Clang 10+ |

---

## Build

Build from the sample folder using CMake directly, or use the PowerShell
helper at the repo root which builds all C samples in one shot.

### CMake directly — Linux

```bash
export XIMEA_SP_PATH=/opt/XIMEA
cd samples/acquisition/capture-10-images/c
cmake -B .cmake-tmp -DXIMEA_SP_PATH="$XIMEA_SP_PATH"
cmake --build .cmake-tmp
```

Binary lands in `.cmake-tmp/build/`.

### CMake directly — Windows (PowerShell)

```powershell
$env:XIMEA_SP_PATH = "C:\XIMEA"
cd samples\acquisition\capture-10-images\c
cmake -B .cmake-tmp -A x64 -DXIMEA_SP_PATH $env:XIMEA_SP_PATH
cmake --build .cmake-tmp --config Release
```

Binary lands in `.cmake-tmp\build\Release\`.

### Repo helper script — Windows (PowerShell)

Builds all C samples and copies binaries to `build\`:

```powershell
$env:XIMEA_SP_PATH = "C:\XIMEA"
.\build.ps1
```

Output is copied to `build\acquisition-capture-10-images-c\`.

---

## Run

After a direct CMake build:

```bash
# Linux (.cmake-tmp/build/)
.cmake-tmp/build/acquisition-capture-10-images-c

# Windows PowerShell (.cmake-tmp\build\Release\)
.\.cmake-tmp\build\Release\acquisition-capture-10-images-c.exe
```

After `build.ps1`:

```powershell
.\build\acquisition-capture-10-images-c\acquisition-capture-10-images-c.exe
```

The sample takes no arguments. It always opens camera index 0, sets exposure
to 100 ms, and captures 10 frames.

---

## Expected output

```
Found 1 camera(s), opening index 0
Exposure: 100000 us (100 ms)
Capturing 10 frames
Frame 1/10: 1280x1024 nframe=1 first_byte=42
Frame 2/10: 1280x1024 nframe=2 first_byte=41
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

Exact width, height, and first_byte values depend on your camera model and scene.

---

## Known limitations / caveats

- No CLI parameters — frame count, exposure, and camera index are compile-time
  constants (`frameCount`, `exposureUs`, `grabTimeoutMs` in `main.c`).
- Only camera index 0 is opened. If multiple cameras are connected, the first
  one detected by the SDK is used.
- The per-frame grab timeout is fixed at 5000 ms. Because exposure is 100 ms,
  5000 ms provides a comfortable margin, but very slow hosts or USB hubs may
  still time out under load.
- `xiSetParamInt(XI_PRM_EXPOSURE)` takes microseconds. 100 ms = 100 000 µs.
  Not all camera models support every exposure value; the SDK will return a
  non-OK status if the value is out of range for the connected model.
- On Windows, the executable runs against `xiapi64.dll` from the XIMEA Software
  Package under `%XIMEA_SP_PATH%\API\xiAPI\`. Ensure that directory is on your
  `PATH`, or start the program from an environment where it is reachable.
- On Linux, `RPATH` is set at build time to the directory where `libm3api` was
  found. That may be inside `$XIMEA_SP_PATH/lib64` or `$XIMEA_SP_PATH/lib`, or in
  a standard system library path if the XIMEA runtime is installed globally.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/allprod/XIMEA_API_Manual)
- [XIMEA Linux Software Package](https://www.ximea.com/support/wiki/allprod/XIMEA_Linux_Software_Package)
- [XIMEA Windows Software Package](https://www.ximea.com/support/wiki/allprod/XIMEA_Windows_Software_Package)
