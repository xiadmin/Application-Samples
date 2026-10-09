# capture-10-concatenated-images-to-file — GenTL C++ sample

Captures 2 GenTL transport buffers that each hold 5 concatenated images, splits every buffer into its images and saves them as TIFF files.

| Item | Value |
|------|-------|
| Category | GenTL / concatenated images |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | GenTL |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | A XIMEA camera that supports `XI_PRM_CONCAT_IMG_MODE` and `XI_PRM_GENTL_DATASTREAM_ENABLED` |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads), including the examples |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | XIMEA GenTL helpers (`xiGenTL`), built from the SP sources by `ximea_find_gentl()`; libtiff 4.7.1 (central dependency, fetched and built at configure time) |

---

## Build

### Linux

```bash
cd samples/gentl/cross-platform/capture-10-concatenated-images-to-file
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\gentl\cross-platform\capture-10-concatenated-images-to-file
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows. If the GenTL helpers are not found, CMake prints a warning and skips the sample.

---

## Run

```bash
# Linux
.cmake-tmp/build/gentl-cross-platform-capture-10-concatenated-images-to-file --output-dir images

# Windows PowerShell
.\.cmake-tmp\build\gentl-cross-platform-capture-10-concatenated-images-to-file.exe --output-dir images
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--output-dir <dir>` | current directory | Directory for `buffer1_image1.tiff` … `buffer2_image5.tiff`; created if missing |

---

## Expected output

```text
Opening first camera
5 images per buffer, payload 6553600 bytes, image offset 1310720 bytes
Preparing 2 acquisition buffers
Starting acquisition
Saved images/buffer1_image1.tiff
...
Saved images/buffer2_image5.tiff
Done
```

Without a camera the sample prints `Error: no XIMEA cameras detected` and exits with a failure code.

---

## Known limitations / caveats

- With `XI_PRM_CONCAT_IMG_MODE` on, the camera packs `XI_PRM_CONCAT_IMG_COUNT` frames into one transport buffer, which lowers the per-buffer overhead at high frame rates. Each image starts `XI_PRM_CONCAT_IMG_TRANSPORT_IMG_OFFSET` bytes after the previous one.
- The sample checks that the reported offset and image size fit the 5 images into the payload before it reads them, and stops with an error otherwise.
- Images are saved as 8-bit monochrome TIFFs from RAW8 transport data. On a color sensor this is the undemosaiced Bayer mosaic.
- The SDK GenTL helper headers declare `using namespace std;` at global scope, so `xiApi.h` is included before them (see the comment in `main.cpp`).
- The sample always opens camera index 0. Existing files with the same names in the output directory are overwritten.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [GenICam GenTL standard (EMVA)](https://www.emva.org/standards-technology/genicam/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
