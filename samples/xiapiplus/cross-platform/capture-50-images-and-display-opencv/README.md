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
| Libraries | OpenCV 4 (core, imgproc, highgui, videoio), found with `find_package(OpenCV)`; see [Third-party libraries](#third-party-libraries) |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |

---

## Third-party libraries

This sample uses **OpenCV 4**, which is installed rather than built with the sample. Nothing is downloaded unless you ask for it; a configure that finds no OpenCV prints a warning and skips this sample.

| `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` | What happens |
|---|---|
| not set (default) | CMake uses a copy you installed yourself through `find_package(OpenCV)`; nothing is downloaded. On Debian/Ubuntu, run `sudo apt install libopencv-dev`. On Windows, run `dependencies\install-opencv-windows.ps1` from the repository root: it downloads the OpenCV installer pinned in the script, accepts it only if its SHA-256 matches the hash pinned there, unpacks it into `C:\tools\opencv` and sets `OpenCV_DIR` and `PATH` for your user. For a copy you unpacked yourself, pass its CMake package folder: `-DOpenCV_DIR=<your unpacked OpenCV>/build`. |
| `1` | On Windows, if no OpenCV is found, CMake runs `dependencies/install-opencv-windows.ps1` with its pinned version and hash: the installer is downloaded into `dependencies/cache/downloads/`, used only if its SHA-256 matches, and unpacked into `C:\tools\opencv`, where the configure then finds it. Open a new terminal before running the sample so that the OpenCV `bin` folder the script adds to `PATH` is picked up. On Linux nothing is downloaded; install `libopencv-dev` as above. |

To let this and every other sample download what it needs, set `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES` to `1` (`ON`, `TRUE` and `YES` work too) before you build; the commands below then work unchanged. Unset it to go back to your own copies. See [dependencies/README.md](../../../../dependencies/README.md) for the central dependency files.

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
cd samples/xiapiplus/cross-platform/capture-50-images-and-display-opencv
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\capture-50-images-and-display-opencv
cmake -B .cmake-tmp -A x64
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
- The sample always opens the first camera.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [OpenCV documentation](https://docs.opencv.org/4.x/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
