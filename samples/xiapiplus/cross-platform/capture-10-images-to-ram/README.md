# capture-10-images-to-ram — C++ sample

Captures 10 images into application-owned RAM.

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

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/capture-10-images-to-ram
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-10-images-to-ram
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

```bash
# Linux
.cmake-tmp/build/xiapiplus-cross-platform-capture-10-images-to-ram

# Windows PowerShell
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-10-images-to-ram.exe
```

---

## Expected output

The sample captures 10 RAW8 frames, copies each one into a `std::vector` owned by the application, closes the camera, and then prints per-frame metadata from RAM.

```text
Opening camera index 0
Camera closed; reading retained images from RAM
Frame 1/10: 1280x1024 nframe=1 first_byte=42
...
Frame 10/10: 1280x1024 nframe=10 first_byte=39
Done
```

---

## Known limitations / caveats

- A connected XIMEA camera and installed SDK runtime are required; the sample exits with an error if no camera is detected.
- The sample always opens camera index 0 and uses a fixed 10 ms exposure.
- Frames are copied into `std::vector<unsigned char>` instead of `xiAPIplus_Image::CopyTo()`: the buffer that `CopyTo()` allocates is not marked as a copy, so the `xiAPIplus_Image` destructor never frees it.
- The stored pixels include any line padding the driver adds.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [xiAPI buffer policy](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_BUFFER_POLICY)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
