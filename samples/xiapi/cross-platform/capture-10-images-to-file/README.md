# capture-10-images-to-file — C sample

Captures 10 stabilized frames from the first available XIMEA camera and saves them as TIFF files.

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
| Network | Required on the first clean build to download libtiff 4.7.1 |

CMake downloads libtiff 4.7.1 automatically into `.cmake-tmp` during the first configure.
The download is controlled by `dependencies/Dependencies.cmake` at the repository root.
Reconfiguring and building in the same populated build tree works without network access.
A clean build (empty `.cmake-tmp`) needs network access or a pre-populated FetchContent cache.

---

## Build

### Linux

```bash
cd samples/xiapi/cross-platform/capture-10-images-to-file
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapi\cross-platform\capture-10-images-to-file
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

## Run

Run from the directory where the TIFF files should be created:

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-capture-10-images-to-file

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-capture-10-images-to-file.exe
```

---

## Expected behavior

The sample enables automatic exposure/gain, discards 10 warm-up frames, and saves the next 10 frames as `image000.tif` through `image009.tif`. Color cameras use RGB24 output; monochrome cameras use MONO8.

```text
Opening camera index 0
Saved frame 1/10 to image000.tif
...
Saved frame 10/10 to image009.tif
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required.
- The sample always opens camera index 0 and captures a fixed count of 10 frames after 10 auto-exposure warm-up frames.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- Output files are uncompressed baseline TIFF (no compression) for maximum reader compatibility.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [xiAPI buffer policy](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_BUFFER_POLICY)
