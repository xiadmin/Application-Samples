# error-handling — C sample

Shows how to detect and recover from an xiAPI error such as XI_TIMEOUT.

| Item | Value |
|------|-------|
| Category | Error handling |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI |
| Language | C |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |

---

## Build

### Linux

```bash
cd samples/xiapi/cross-platform/error-handling
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapi\cross-platform\error-handling
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-error-handling

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-error-handling.exe
```

---

## Expected output

The sample first calls the API without an open camera, which fails with `XI_INVALID_HANDLE` and needs no hardware. It then opens the camera with a 100 ms exposure and waits only 1 ms for the first frame. The resulting `XI_TIMEOUT` is caught and handled by retrying with a 5000 ms timeout. Any other error ends the sample with a non-zero exit code; on success it exits with 0.

```text
Caught XI_INVALID_HANDLE (1) from a call without an open camera
Opening camera index 0
Caught XI_TIMEOUT (10) with a 1 ms timeout; retrying with 5000 ms
Got frame 1280x1024 nframe=1
Done
```

Without a camera, only the first line is printed, followed by `Error: no XIMEA cameras detected` on stderr and exit code 1.

---

## Known limitations / caveats

- Fatal errors use the `CE` macro, which reports the failing call and its code on stderr; `XI_TIMEOUT` is checked by hand because it is recoverable.
- The `XI_TIMEOUT` path depends on the first frame taking longer than 1 ms. If a frame does arrive in time, the sample prints `A frame arrived within 1 ms; no timeout this time` and continues.
- The sample always opens camera index 0.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
