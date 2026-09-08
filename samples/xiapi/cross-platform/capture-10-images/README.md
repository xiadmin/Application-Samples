# capture-10-images — C sample

Captures 10 frames from the first available XIMEA camera and prints basic information about each frame.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| API type | xiAPI |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |

---

## Build

### Linux

```bash
cd samples/xiapi/cross-platform/capture-10-images
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapi\cross-platform\capture-10-images
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

## Run

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-capture-10-images

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-capture-10-images.exe
```

---

## Expected output

```text
Opening camera index 0
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required.
- The sample always opens camera index 0 and uses a fixed count of 10 frames, a 10 ms exposure, and a 5-second image timeout.
- Returned pixel data is only inspected immediately; this sample does not retain frames in application memory.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
