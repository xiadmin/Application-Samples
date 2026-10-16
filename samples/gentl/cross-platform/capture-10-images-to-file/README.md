# capture-10-images-to-file — GenTL C++ sample

Loads the XIMEA GenTL producer (`.cti`) at run time, opens the first camera through the GenTL module hierarchy, captures 10 images into application buffers and saves them as TIFF files.

| Item | Value |
|------|-------|
| Category | GenTL / low-level acquisition |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | GenTL |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA camera supported by the XIMEA GenTL producer |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads), including the GenTL producer and the examples (`GenTL_v1_5.h`) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | libtiff (central dependency in `dependencies/Dependencies.cmake`, fetched and built at configure time) |

---

## Build

### Linux

```bash
cd samples/gentl/cross-platform/capture-10-images-to-file
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\gentl\cross-platform\capture-10-images-to-file
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows. If the GenTL headers are not found, CMake prints a warning and skips the sample.

---

## Run

```bash
# Linux
.cmake-tmp/build/gentl-cross-platform-capture-10-images-to-file --output-dir images

# Windows PowerShell
.\.cmake-tmp\build\gentl-cross-platform-capture-10-images-to-file.exe --output-dir images
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--cti <file>` | `ximea.gentl2.cti`, then `ximea.gentl.cti`, from `GENICAM_GENTL64_PATH` (`GENICAM_GENTL32_PATH` in 32-bit builds) | GenTL producer to load |
| `--output-dir <dir>` | current directory | Directory for `image_00.tiff` … `image_09.tiff`; created if missing |

---

## Expected output

```text
Loading GenTL producer C:\XIMEA\GenTL Producer\ximea.gentl2.cti
Opening device <device id>
Resolution 1280x1024, payload 1310720 bytes
Acquiring 10 images
Saved images/image_00.tiff
...
Saved images/image_09.tiff
Done
```

Without a camera the sample prints `Error: the GenTL producer found no camera device` and exits with a failure code.

---

## Known limitations / caveats

- The sample does not link xiAPI; it talks to the producer only through the GenTL C interface that it resolves from the `.cti` at run time.
- Width, height, payload size, exposure and the acquisition start/stop commands are read and written at fixed XIMEA register addresses. The register map is chosen from the producer file name: `ximea.gentl2.cti` uses the current map, any other name the legacy one. A generic GenTL consumer reads these addresses from the device's GenICam XML instead.
- The images are saved as 8-bit monochrome TIFFs. The sample stops with an error when the payload is smaller than one byte per pixel; other pixel formats are not converted.
- A frame that does not arrive within 2 s is reported and skipped; the sample then exits with a failure code.
- Existing files with the same names in the output directory are overwritten.

---

## Links

- [GenICam GenTL standard (EMVA)](https://www.emva.org/standards-technology/genicam/)
- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
