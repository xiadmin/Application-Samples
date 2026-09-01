# capture-10-images-to-ram — C sample

Captures 10 RAW8 frames into application-owned RAM and accesses them after the camera is closed.

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
| XIMEA SP | [Approved version in DEPENDENCIES.md](../../../../DEPENDENCIES.md#2-approved-dependency-table) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |

---

## Build

### Linux

```bash
cd samples/xiapi/cross-platform/capture-10-images-to-ram
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapi\cross-platform\capture-10-images-to-ram
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

## Run

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-capture-10-images-to-ram

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-capture-10-images-to-ram.exe
```

---

## Expected behavior

The sample allocates one distinct destination for each frame, captures with `XI_BP_SAFE`, stops acquisition, closes the camera, and then reads the retained frame information from application-owned memory.

```text
Opening camera index 0
Camera closed; reading retained images from RAM
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required.
- The sample always opens camera index 0 and uses RAW8 output, a fixed count of 10 frames, a 10 ms exposure, and a 5-second image timeout.
- RAM usage is approximately 10 times the value reported by `XI_PRM_IMAGE_PAYLOAD_SIZE`.
- The allocated buffers are released after the retained frame information is printed.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [xiAPI buffer policy](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_BUFFER_POLICY)
