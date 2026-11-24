# capture-10-images-to-file — C++ sample

Captures 10 images and saves them as TIFF files using libtiff.

| Item | Value |
|------|-------|
| Category | Basic acquisition / image capture |
| OS platform | Cross-Platform |
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
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | libtiff (central dependency in `dependencies/Dependencies.cmake`, downloaded and SHA-256 checked, or found installed; see [Third-party libraries](#third-party-libraries)) |
| Network | With `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` set, required only until `dependencies/cache/downloads/` at the repository root holds the libtiff archive; later builds reuse the cached archive. Not set, no network is used. Pass `-DAPPLICATION_SAMPLES_OFFLINE=ON` to fail fast instead of reaching for the network. |

---

## Third-party libraries

This sample uses **libtiff**. Nothing is downloaded unless you ask for it; a configure that finds no installed copy stops and explains both ways.

| `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` | What happens |
|---|---|
| not set (default) | CMake uses a copy you installed yourself (`sudo apt install libtiff-dev`, `brew install libtiff`, or `vcpkg install tiff`) through `find_package(TIFF)`; nothing is downloaded. A copy you build and install into `dependencies/local/cmake/<library>/` at the repository root is found with no extra options; point `CMAKE_PREFIX_PATH` at an install anywhere else outside the default locations. |
| `1` | CMake downloads the archive named in `dependencies/Dependencies.cmake` into `dependencies/cache/downloads/` and builds it with the sample. The archive, new or cached, is used only if its SHA-256 matches the hash recorded there. |

To let this and every other sample download what it needs, set `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` to `1` (`ON`, `TRUE` and `YES` work too) before you build or run; the commands below then work unchanged. Unset it to go back to your own copies.

```bash
export AUTOMATICALLY_DOWNLOAD_DEPENDENCIES=1   # Linux/macOS
```

```powershell
$env:AUTOMATICALLY_DOWNLOAD_DEPENDENCIES = '1'   # Windows PowerShell
```

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/capture-10-images-to-file
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-10-images-to-file
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

Run from the repository root. Output files are written to the current directory (the sample folder in the commands below).

```bash
# Linux
cd samples/xiapiplus/cross-platform/capture-10-images-to-file
.cmake-tmp/build/xiapiplus-cross-platform-capture-10-images-to-file

# Windows PowerShell
cd samples\xiapiplus\cross-platform\capture-10-images-to-file
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-10-images-to-file.exe
```

---

## Expected output

The sample enables automatic exposure/gain, discards 10 warm-up frames, and saves the next 10 frames as `image000.tif` through `image009.tif`. Color cameras use RGB24 output with automatic white balance; monochrome cameras use MONO8. `WriteImage()` from xiAPIplus prints its own `Saving image ...` line before each confirmation.

```text
Opening camera index 0
Saving image 1280x1024 to file:image000.tif
Saved frame 1/10 to image000.tif
...
Saving image 1280x1024 to file:image009.tif
Saved frame 10/10 to image009.tif
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- TIFF writing uses `WriteImage()` from the SDK's `xiAPIplus_tiff.cpp`, which `ximea_find_sdk()` compiles into `XIMEA::xiAPIplus` only when `application_samples_use(libtiff)` runs first.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [xiAPI buffer policy](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_BUFFER_POLICY)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
