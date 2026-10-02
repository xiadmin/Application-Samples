# capture-10-images-to-dng — C++ sample

Captures 10 raw images and saves them as DNG files using the XIMEA DNG Store library.

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
| XIMEA SP | [4.33](https://www.ximea.com/software-downloads) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| DNG Store | The "DNG Store" component of the XIMEA SP (installed by default) |

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/capture-10-images-to-dng
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-10-images-to-dng
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

`xiapi_dng_store.dll` is copied next to the executable on Windows. Run from the directory where the DNG files should be created:

```bash
# Linux
.cmake-tmp/build/xiapiplus-cross-platform-capture-10-images-to-dng

# Windows PowerShell
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-10-images-to-dng.exe
```

---

## Expected output

The sample captures 10 RAW8 frames, Bayer data on a color camera, and saves each as `image000.dng` through `image009.dng`. Camera metadata (model, serial number, color filter array, white balance) is read once with `xidngFillMetadataFromCameraParams` and written into every file. DNG files open in raw converters such as Adobe Lightroom, darktable or RawTherapee.

```text
Opening camera index 0
Saved frame 1/10 (1280x1024) to image000.dng
...
Saved frame 10/10 (1280x1024) to image009.dng
Done
```

---

## Known limitations / caveats

- If the DNG Store component is missing, CMake prints a warning and configures no target for this sample. `ximea_find_dng_store()` looks in `<XIMEA_SP_PATH>/DNG Store` on Windows, `/opt/XIMEA` on Linux and `/Library/Frameworks/m3api.framework` on macOS. Builds are verified on Windows; the Linux and macOS locations come from the SP installers.
- xiAPIplus has no DNG wrapper; the sample passes `GetCameraHandle()` and `GetXI_IMG()` to the C DNG Store API and turns its return codes into `xiAPIplus_Exception`.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Existing `image000.dng` through `image009.dng` files in the current directory are overwritten.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
