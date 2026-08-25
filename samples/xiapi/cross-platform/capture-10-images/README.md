# capture-10-images — C sample

Captures 10 frames from the first available XIMEA camera in print, TIFF, or application-owned RAM mode.

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
| XIMEA SDK | 4.32+ |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |

No external TIFF library is required. The sample links the repository-local baseline TIFF writer in `libs/tiff-writer/`.

---

## Build

### Linux

```bash
cd samples/xiapi/cross-platform/capture-10-images
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

The binary lands in `.cmake-tmp/build/`.

### Windows (PowerShell)

```powershell
cd samples\xiapi\cross-platform\capture-10-images
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp\build\`.

## Run

Print frame metadata and the first byte. This is the default mode:

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-capture-10-images --mode print

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-capture-10-images.exe --mode print
```

Capture 10 stabilized frames as `image000.tif` through `image009.tif` in the current directory:

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-capture-10-images --mode tiff

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-capture-10-images.exe --mode tiff
```

Capture into 10 application-owned buffers, close the camera, and then read the retained frame metadata and first bytes:

```bash
# Linux
.cmake-tmp/build/xiapi-cross-platform-capture-10-images --mode ram

# Windows PowerShell
.\.cmake-tmp\build\xiapi-cross-platform-capture-10-images.exe --mode ram
```

Run with `--help` to list the available modes.

---

## Expected behavior

All modes report the detected camera count and the fixed 10 ms exposure.

- `print`: captures 10 frames and prints dimensions, frame number, and first byte for each frame.
- `tiff`: enables automatic exposure/gain, discards 10 warm-up frames, and saves the next 10 frames as uncompressed TIFF files. Color cameras use RGB24 output; monochrome cameras use MONO8.
- `ram`: uses `XI_BP_SAFE` with application-owned RAW8 buffers, closes the camera, and then prints the retained frame information from RAM.

Example print-mode output:

```text
Found 1 camera(s), opening index 0
Exposure: 10000 us (10 ms)
Capturing 10 frames in print mode
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required for acquisition; `--help` works without a camera.
- The sample always opens camera index 0 and uses a fixed count of 10 frames with a fixed 10 ms initial exposure.
- TIFF mode overwrites existing `image000.tif` through `image009.tif` files in the current directory.
- RAM usage is approximately 10 times the value reported by `XI_PRM_IMAGE_PAYLOAD_SIZE`.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [xiAPI buffer policy](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_BUFFER_POLICY)
