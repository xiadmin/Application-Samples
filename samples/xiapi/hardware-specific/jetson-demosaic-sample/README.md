# Jetson Demosaic Sample — C++/CUDA sample

GPU demosaicing sample for Nvidia Jetson that bypasses xiAPI CPU processing for color cameras, renders frames, saves TIFF output, and prints processing timing statistics.

| Item | Value |
|------|-------|
| Category | Color / Bayer / color correction |
| API type | xiAPI |

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux on Nvidia Jetson |
| Hardware | Nvidia Jetson kit with a supported XIMEA color camera |
| XIMEA SP | [Approved version in DEPENDENCIES.md](../../../../DEPENDENCIES.md#2-approved-dependency-table) |
| CMake | 3.16 or newer |
| Compiler | GCC/G++ with C++17 support and CUDA `nvcc` |
| Libraries | CUDA Toolkit, NPP, OpenCV built with CUDA/OpenGL support |

---

## Build

This sample is hardware-specific. The CMake project intentionally skips itself on non-Linux or non-aarch64 systems.

### CMake directly — Jetson

```bash
cd samples/xiapi/hardware-specific/jetson-demosaic-sample
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

Binary lands in `.cmake-tmp/build/`.

### Makefile — Jetson

The legacy Makefile is still available for direct Jetson builds:

```bash
cd samples/xiapi/hardware-specific/jetson-demosaic-sample
make
```

---

## Run

After a direct CMake build:

```bash
.cmake-tmp/build/xiapi-hardware-specific-jetson-demosaic-sample
```

After a Makefile build:

```bash
./jetson_sample
```

---

## Expected output

The sample opens a camera, captures frames into CUDA-accessible memory, runs depacking/demosaicing, renders the image, saves TIFF output, and prints per-stage timing statistics.

---

## Known limitations / caveats

- Jetson-specific: this sample depends on Nvidia Jetson hardware, CUDA, NPP, and OpenCV CUDA/OpenGL support.
- Non-Jetson CMake configure runs skip this sample instead of failing the whole repository build.
- The source currently preserves the original Jetson sample style and has not yet been modernized.

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
