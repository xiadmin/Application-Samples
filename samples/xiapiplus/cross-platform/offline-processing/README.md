# offline-processing — C++ sample

Captures 10 RAW8 images, converts them to RGB24 with xiAPI offline processing after acquisition, and saves them as TIFF files.

| Item | Value |
|------|-------|
| Category | Image processing |
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
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | libtiff 4.7.1 (central dependency, fetched and built at configure time) |
| Network | Required only until `.deps/downloads/` at the repository root holds libtiff 4.7.1; later builds reuse the cached archive. Pass `-DAPPLICATION_SAMPLES_OFFLINE=ON` to fail fast instead of reaching for the network. |

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/offline-processing
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\offline-processing
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

Run from the directory where the TIFF files should be created:

```bash
# Linux
.cmake-tmp/build/xiapiplus-cross-platform-offline-processing

# Windows PowerShell
.\.cmake-tmp\build\xiapiplus-cross-platform-offline-processing.exe
```

---

## Expected output

The sample captures 10 RAW8 frames into RAM. It reads the camera's API context list (`XI_PRM_API_CONTEXT_LIST`, every setting the frames were captured with) and then closes the camera. The xiAPI offline processing (`xiProc`) then converts each stored frame using that context and saves it as `image000.tif` through `image009.tif`. Color cameras produce RGB24 with the white balance coefficients recorded in the context; monochrome cameras produce MONO8.

`WriteImage()` from xiAPIplus prints its own `Saving image ...` line before each confirmation.

```text
Opening camera index 0
Camera closed; processing 10 stored frames offline
Saving image 1280x1024 to file:image000.tif
Processed frame 1/10 (1280x1024) to image000.tif
...
Saving image 1280x1024 to file:image009.tif
Processed frame 10/10 (1280x1024) to image009.tif
Done
```

---

## Known limitations / caveats

- Uses `xiAPIplus_ImageProcessing`, which opens the processing in its constructor and closes it in its destructor.
- `xiAPIplus_ImageProcessing::GetImage()` returns a `malloc`'d copy that is not marked as one; the sample calls `SetCopy(true)` so the `xiAPIplus_Image` destructor frees it.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Existing `image000.tif` through `image009.tif` files in the current directory are overwritten.
- The white balance coefficients (1.05 / 1.0 / 2.10) are example values from the original sample; adjust them for your light source.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
