# capture-10-images — C++ sample

Captures 10 frames from the first available XIMEA camera and prints per-frame metadata.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPIplus |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Libraries | None |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/capture-10-images
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-10-images
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

After a direct CMake build, from the repository root:

```bash
# Linux
cd samples/xiapiplus/cross-platform/capture-10-images
.cmake-tmp/build/xiapiplus-cross-platform-capture-10-images

# Windows PowerShell
cd samples\xiapiplus\cross-platform\capture-10-images
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-10-images.exe
```

---

## Expected output

The sample reports how many cameras it found, sets a 100 ms exposure and prints the size, frame number and first pixel byte of each of the 10 frames.

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

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
