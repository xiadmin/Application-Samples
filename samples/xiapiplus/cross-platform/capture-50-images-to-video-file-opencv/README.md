# capture-50-images-to-video-file-opencv — C++ sample

Captures 50 grayscale images with the xiAPIplus OpenCV wrapper, shows them in a window and records them to a Motion JPEG AVI file.

| Item | Value |
|------|-------|
| Category | Video recording / OpenCV integration |
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
cd samples/xiapiplus/cross-platform/capture-50-images-to-video-file-opencv
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-50-images-to-video-file-opencv
cmake -B .cmake-tmp -A x64 -DOpenCV_DIR=C:/opencv/build
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows. If OpenCV or the wrapper is not found, CMake prints a warning and skips the sample.

---

## Run

Run from the repository root. Output files are written to the current directory (the sample folder in the commands below).

```bash
# Linux
cd samples/xiapiplus/cross-platform/capture-50-images-to-video-file-opencv
.cmake-tmp/build/xiapiplus-cross-platform-capture-50-images-to-video-file-opencv --output capture.avi

# Windows PowerShell (OpenCV's bin folder must be on PATH)
cd samples\xiapiplus\cross-platform\capture-50-images-to-video-file-opencv
.\.cmake-tmp\build\xiapiplus-cross-platform-capture-50-images-to-video-file-opencv.exe --output capture.avi
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--output <file.avi>` | `capture.avi` in the current directory | Video file to write; an existing file is overwritten |

---

## Expected output

A window titled `XIMEA camera` shows the live image while the frames are recorded:

```text
Opening first camera
Starting acquisition
Recorded frame 1/50
...
Recorded frame 50/50
Saved capture.avi
Done
```

Without a camera the sample prints `Error: no XIMEA cameras detected` and exits with a failure code.

---

## Known limitations / caveats

- The camera is switched to `XI_MONO8`, so the video is grayscale even on color cameras.
- The video uses Motion JPEG (`MJPG`) in an AVI container at a nominal 25 fps. OpenCV encodes this format itself, so no FFmpeg or system codec is needed. The legacy sample used Microsoft RLE (`MRLE`), which OpenCV cannot write on every platform.
- The nominal frame rate is stored in the file; capture timing is not, so playback speed does not match the real exposure rate.
- `xiApiPlusOcv.hpp` contains its own copy of the xiAPIplus classes, so `XIMEA::xiAPIplusOpenCV` must not be combined with `XIMEA::xiAPIplus` in one target.
- `cv::imshow()` needs a display; the sample does not run in a headless session.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [OpenCV VideoWriter](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
