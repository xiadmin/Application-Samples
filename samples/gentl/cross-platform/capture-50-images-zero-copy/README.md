# capture-50-images-zero-copy — GenTL C++ sample

Configures the camera with xiAPI, then captures 50 images through the camera's GenTL data stream into buffers allocated by the application (or by the producer with `--auto-buffers`) and prints the metadata of each frame.

| Item | Value |
|------|-------|
| Category | GenTL / zero-copy acquisition |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | GenTL |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | A XIMEA camera that supports `XI_PRM_GENTL_DATASTREAM_ENABLED` |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads), including the examples |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | XIMEA GenTL helpers (`GenTLStream`, `ImgBuffer`), built from the SP sources by `ximea_find_gentl()` |

---

## Build

### Linux

```bash
cd samples/gentl/cross-platform/capture-50-images-zero-copy
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\gentl\cross-platform\capture-50-images-zero-copy
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows. If the GenTL helpers are not found, CMake prints a warning and skips the sample.

---

## Run

```bash
# Linux
.cmake-tmp/build/gentl-cross-platform-capture-50-images-zero-copy

# Windows PowerShell
.\.cmake-tmp\build\gentl-cross-platform-capture-50-images-zero-copy.exe
```

| Option | Meaning |
|--------|---------|
| `--auto-buffers` | Let the GenTL producer allocate the 10 acquisition buffers (`DSAllocAndAnnounceBuffer`). By default the application allocates 4 KiB-aligned buffers and announces them (`DSAnnounceBuffer`), so the camera writes frames straight into application memory. |

---

## Expected output

```text
Opening first camera
Announcing 10 buffers of 1310720 bytes, allocated by the application
Starting acquisition
Frame   1  1280x1024  exposure 10.000 ms  timestamp 12.345678 s  first byte 0x3a
...
Frame  50  1280x1024  exposure 10.000 ms  timestamp 12.845678 s  first byte 0x3b
Done
```

Without a camera the sample prints `Error: no XIMEA cameras detected` and exits with a failure code.

---

## Known limitations / caveats

- xiAPI is used only to open and configure the camera. After `XI_PRM_GENTL_DATASTREAM_ENABLED` is switched on, buffers and acquisition run through the GenTL data stream from `XI_PRM_GENTL_DATASTREAM_CONTEXT`, and `xiGetImage()` is not used.
- The image format is `XI_FRM_TRANSPORT_DATA` on top of RAW8, so the buffers hold the data exactly as the camera sends it, without any processing on the host.
- The SDK GenTL helper headers declare `using namespace std;` at global scope. `xiApi.h` is therefore included before them, so `<windows.h>` is parsed before `std::byte` becomes ambiguous with the Windows `byte` type.
- The sample always opens camera index 0, and a frame that does not arrive within 5 s ends the sample with an error.
- A buffer whose transfer was interrupted is reported as a warning and queued again.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [GenICam GenTL standard (EMVA)](https://www.emva.org/standards-technology/genicam/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
