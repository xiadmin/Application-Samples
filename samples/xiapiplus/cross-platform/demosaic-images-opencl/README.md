# demosaic-images-opencl — C++ sample

Captures RAW8 Bayer frames from a color camera, demosaics them with a Malvar-He-Cutler OpenCL kernel on the GPU, saves them as RGB TIFF files and prints the kernel and transfer times.

| Item | Value |
|------|-------|
| Category | GPU processing / OpenCL integration |
| OS platform | Cross-platform |
| Hardware platform | XIMEA color camera; OpenCL GPU or other OpenCL device |
| API type | xiAPIplus |
| Language | C++ |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11, Linux (Ubuntu 20.04+) or macOS |
| Hardware | A XIMEA color camera with a Bayer filter (RGGB, GRBG, GBRG or BGGR) and an OpenCL 1.2 device, preferably a GPU |
| XIMEA SDK | Latest beta of the [XIMEA Software Package](https://www.ximea.com/software-downloads) |
| OpenCL | Headers and ICD loader at build time; an OpenCL runtime (part of the GPU driver) at run time |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ (C++17) |
| Libraries | libtiff (central dependency in `dependencies/Dependencies.cmake`, fetched and built at configure time) |

CMake finds OpenCL with `find_package(OpenCL)`, and the sample is skipped with a warning when it is missing:

- **Windows:** the GPU vendor's SDK provides the headers and `OpenCL.lib`; the CUDA Toolkit does so for NVIDIA. The runtime (`OpenCL.dll`) comes with the GPU driver.
- **Linux:** `sudo apt install opencl-headers ocl-icd-opencl-dev`, plus the vendor's OpenCL runtime (for example `intel-opencl-icd` or the NVIDIA driver).
- **macOS:** the OpenCL framework is part of the system.

---

## Build

### Linux

```bash
cd samples/xiapiplus/cross-platform/demosaic-images-opencl
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

### Windows (PowerShell)

```powershell
cd samples\xiapiplus\cross-platform\demosaic-images-opencl
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

The binary lands in `.cmake-tmp/build/` on Linux and `.cmake-tmp\build\` on Windows.

---

## Run

Run from the repository root. Output files are written to `frames/` inside the current directory (the sample folder in the commands below).

```bash
# Linux
cd samples/xiapiplus/cross-platform/demosaic-images-opencl
.cmake-tmp/build/xiapiplus-cross-platform-demosaic-images-opencl --output-dir frames

# Windows PowerShell
cd samples\xiapiplus\cross-platform\demosaic-images-opencl
.\.cmake-tmp\build\xiapiplus-cross-platform-demosaic-images-opencl.exe --output-dir frames
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--frame-count <n>` | 10 | Frames to demosaic and save |
| `--exposure-us <us>` | 10000 | Exposure time in microseconds |
| `--output-dir <dir>` | current directory | Directory for `image_00.tiff`, `image_01.tiff`, ... |

---

## Expected output

```text
Opening first camera
Image size 2064x1544, color filter array 1, white balance R 1.28 G 1 B 3.25
OpenCL device: NVIDIA RTX A4000
Frame 1/10: kernel 0.214 ms, upload + kernel + download 3.902 ms, saved frames/image_00.tiff
...
Average: kernel 0.209 ms, upload + kernel + download 3.517 ms
Done
```

The kernel time is measured by the OpenCL device with event profiling. The second time also covers copying the RAW8 frame to the device and the RGB result back.

---

## Known limitations / caveats

- **Color camera only:** a monochrome camera, or a filter array other than the four Bayer layouts (for example polarization or CMYG sensors), is reported as an error.
- **Bayer layout:** the sample reads `XI_PRM_COLOR_FILTER_ARRAY` and passes the position of the red pixel to the kernel, so all four Bayer layouts are handled.
- **White balance:** xiAPI does not apply white balance to RAW8 data. The kernel multiplies the channels by the camera's current `XI_PRM_WB_KR/KG/KB` coefficients. There is no color correction matrix, gamma or sharpening, so the colors differ from xiAPI's own RGB output.
- **Edges:** pixels outside the frame are mirrored by an even distance, which keeps the Bayer color, so the border pixels are demosaiced as well.
- **Device selection:** the first GPU of the first OpenCL platform that has one, otherwise any OpenCL device (for example a CPU runtime).
- The kernel follows Morgan McGuire's GPU formulation of the Malvar-He-Cutler filter, as in the internal `xiSample_opencl_debayer`. The OpenGL/GLUT preview and the OpenCV CUDA path of that sample are not part of this port. The kernel source is compiled into the executable, so no `.cl` file is needed at run time.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XI_PRM_COLOR_FILTER_ARRAY](https://www.ximea.com/support/wiki/apis/XiAPI_Manual#XI_PRM_COLOR_FILTER_ARRAY-or-cfa)
- [Efficient, High-Quality Bayer Demosaic Filtering on GPUs](https://casual-effects.com/research/McGuire2009Bayer/index.html)
- [OpenCL 1.2 reference](https://registry.khronos.org/OpenCL/sdk/1.2/docs/man/xhtml/)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
