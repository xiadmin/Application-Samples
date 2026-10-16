# capture-50-images-and-display-opencv — C++ sample

Captures 50 images as OpenCV matrices with the xiAPIplus OpenCV wrapper and shows them in a window.

| Item | Value |
|------|-------|
| Category | Display / OpenCV integration |
| OS platform | Cross-platform |
| Hardware platform | Cross-platform |
| API type | xiAPIplus |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) with a desktop session |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads), including the examples (`xiApiPlusOcv.hpp`) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | OpenCV 4 (core, imgproc, highgui, videoio), found with `find_package(OpenCV)` |

Install OpenCV on Debian/Ubuntu with `sudo apt install libopencv-dev`. On Windows, unpack an OpenCV release and pass its CMake package folder: `-DOpenCV_DIR=C:/opencv/build`.

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/capture-50-images-and-display-opencv
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-50-images-and-display-opencv
cmake -B .cmake-tmp -A x64 -DOpenCV_DIR=C:/opencv/build
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows. If OpenCV or the wrapper is not found, CMake prints a warning and skips the sample.

---

## Run

From the repository root:

```bash
# Linux
cd samples/xiapiplus/cross-platform/capture-50-images-and-display-opencv
.cmake-tmp/build/xiapiplus-cross-platform-capture-50-images-and-display-opencv

# Windows PowerShell (OpenCV's bin folder must be on PATH)
cd samples\xiapiplus\cross-platform\capture-50-images-and-display-opencv
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-50-images-and-display-opencv.exe
```

---

## Expected output

A window titled `XIMEA camera` shows the live image for about one second, and the console prints the first pixel of every frame:

```text
Opening first camera
Starting acquisition
Frame 1/50: first pixel 37
...
Frame 50/50: first pixel 38
Done
```

Without a camera the sample prints `Error: no XIMEA cameras detected` and exits with a failure code.

---

## Known limitations / caveats

- `xiApiPlusOcv.hpp` contains its own copy of the xiAPIplus classes. `XIMEA::xiAPIplusOpenCV` therefore links only `XIMEA::xiAPI` and must not be combined with `XIMEA::xiAPIplus` in one target.
- The camera keeps its default image format. 16-bit formats are stretched to the full range with `cv::normalize()` before display, because sensors fill only 10–12 of the 16 bits.
- `cv::imshow()` needs a display; the sample does not run in a headless session.
- The sample always opens camera index 0.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [OpenCV documentation](https://docs.opencv.org/4.x/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
