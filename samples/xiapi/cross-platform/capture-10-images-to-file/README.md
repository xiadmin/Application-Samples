# capture-10-images-to-file — C sample

Captures 10 stabilized frames from the first available XIMEA camera and saves them as TIFF files.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| OS platform | Cross-Platform |
| Hardware platform | Cross-platform |
| API type | xiAPI |
| Language | C |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| Libraries | libtiff (central dependency in `dependencies/Dependencies.cmake`, downloaded and SHA-256 checked, or found installed; see [Third-party libraries](#third-party-libraries)) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |
| Network | With `download`, required only until `.deps/downloads/` at the repository root holds the libtiff source archive; later builds reuse the cached archive. Pass `-DAPPLICATION_SAMPLES_OFFLINE=ON` to fail fast instead of reaching for the network. |

---

## Third-party libraries

This sample uses **libtiff**. Nothing is downloaded until you choose how to provide it; a configure without a choice stops and explains both options.

| Choice | What happens | Configure with |
|--------|--------------|----------------|
| `download` | CMake downloads the archive named in `dependencies/Dependencies.cmake` into `.deps/downloads/` and builds it with the sample. The archive, new or cached, is used only if its SHA-256 matches the hash recorded there. | `cmake -B .cmake-tmp -DAPPLICATION_SAMPLES_DEPENDENCIES=download` |
| `system` | CMake uses a copy you installed yourself (`sudo apt install libtiff-dev`, `brew install libtiff`, or `vcpkg install tiff`) through `find_package(TIFF)`; nothing is downloaded. Point `CMAKE_PREFIX_PATH` at an install outside the default locations. | `cmake -B .cmake-tmp -DAPPLICATION_SAMPLES_DEPENDENCIES=system` |

On Windows, add `-A x64` as in the build commands below.

To answer once for every C/C++ and C# sample, set the environment variable instead; the commands below then work unchanged:

```bash
export APPLICATION_SAMPLES_DEPENDENCIES=download   # Linux/macOS; or system
```

```powershell
$env:APPLICATION_SAMPLES_DEPENDENCIES = "download"   # Windows PowerShell; or "system"
```

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

---

## Run

Output files are written to the current directory (the sample folder in the commands below).

```bash
# Linux
cd samples/xiapi/cross-platform/capture-10-images-to-file
.cmake-tmp/build/xiapi-cross-platform-capture-10-images-to-file

# Windows PowerShell
cd samples\xiapi\cross-platform\capture-10-images-to-file
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
