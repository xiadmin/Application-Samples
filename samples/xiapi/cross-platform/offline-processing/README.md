# offline-processing — C sample

Captures 10 RAW8 images, converts them to RGB24 with xiAPI offline processing after acquisition, and saves them as TIFF files.

| Item | Value |
|------|-------|
| Category | Image processing |
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
cd samples/xiapi/cross-platform/offline-processing
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapi\cross-platform\offline-processing
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

Output files are written to the current directory (the sample folder in the commands below).

```bash
# Linux
cd samples/xiapi/cross-platform/offline-processing
.cmake-tmp/build/xiapi-cross-platform-offline-processing

# Windows PowerShell
cd samples\xiapi\cross-platform\offline-processing
.\.cmake-tmp\build\xiapi-cross-platform-offline-processing.exe
```

---

## Expected output

The sample captures 10 RAW8 frames into RAM. It reads the camera's API context list (`XI_PRM_API_CONTEXT_LIST`, every setting the frames were captured with) and then closes the camera. The xiAPI offline processing (`xiProc`) then converts each stored frame using that context and saves it as `image000.tif` through `image009.tif`. Color cameras produce RGB24 with the white balance coefficients recorded in the context; monochrome cameras produce MONO8.

```text
Opening camera index 0
Camera closed; processing 10 stored frames offline
Processed frame 1/10 (1280x1024) to image000.tif
...
Processed frame 10/10 (1280x1024) to image009.tif
Done
```

---

## Known limitations / caveats

- The context list size comes from `XI_PRM_API_CONTEXT_LIST XI_PRMM_REQ_VAL_BUFFER_SIZE`, so the buffer always fits.
- Each pulled image is owned by the processing until the next pull; the sample writes it to disk before pushing the next frame.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- The white balance coefficients (1.05 / 1.0 / 2.10) are example values; adjust them for your light source.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
