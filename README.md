# Application-Samples

Code samples for XIMEA cameras, showing how to control your camera using the XIMEA APIs. Each sample is self-contained and includes its own README.md file with detailed instructions.

---

## Prerequisites

- [XIMEA Software Package (SP) 4.33](https://www.ximea.com/software-downloads).
- Python 3.11 or newer for Python samples.
- CMake 3.16 or newer plus a C/C++ compiler for xiAPI C/C++ samples.
- .NET SDK 8.0 or newer for xiAPI.NET samples.

On Windows, set `XIMEA_SP_PATH` to the SP installation root (for example `C:\XIMEA`).
On Linux, the SP installs to `/opt/XIMEA` by default.

---

## Dependency management

- No external source code or built library is committed to this repository.
- Every build-file package version belongs in one of the three files under `dependencies/`.
- Sample CMake files may use unversioned `find_package` calls, but all `FetchContent` declarations and package versions belong in `dependencies/Dependencies.cmake`.
- Sample Python requirements must activate `dependencies/python-constraints.txt`, request only packages listed there, and contain no local version specifiers.
- Every NuGet `PackageReference` must be versionless and have a matching `PackageVersion` in `dependencies/Directory.Packages.props`.
- Project-owned source libraries may remain inside a sample and use `add_library`, Python imports, or `ProjectReference` as applicable.
- XIMEA Software Package, Compute Unified Device Architecture (CUDA), NVIDIA Performance Primitives (NPP), and Jetson OpenCV remain system dependencies installed separately.
- The dependency policy test rejects sample-local versions and conflicting exact versions across CMake, Python, and .NET build files.
- The first clean CMake configure for a fetched dependency needs network access. Subsequent builds in the same populated build tree work without network access unless dependencies change.

---

## Repository layout

```text
samples/
  xiapi/                    # C samples using xiAPI, plus xiAPI-based hardware-specific samples
  xiapiplus/                # C++ samples using xiAPIplus
  xiapi-net-csharp/         # C# samples using xiAPI.NET
  xiapi-python/             # Python samples using xiAPI Python bindings
cmake/                      # shared sample CMake support files
dependencies/
  Dependencies.cmake       # central FetchContent declarations
  Directory.Packages.props # central NuGet package version control
  python-constraints.txt   # pinned Python package versions
tests/                      # dependency policy checks
```

Each sample has a README.md explaining what it does and how to use it.

---

## Available samples

| Sample | API / Language | What it shows |
|--------|-----------------|---------------|
| [xiapi/cross-platform/capture-10-images](samples/xiapi/cross-platform/capture-10-images/) | XiAPI (C) | Capture 10 frames and print basic frame information |
| [xiapi/cross-platform/capture-10-images-to-file](samples/xiapi/cross-platform/capture-10-images-to-file/) | XiAPI (C) | Capture 10 stabilized frames and save them as TIFF files |
| [xiapi/cross-platform/capture-10-images-to-ram](samples/xiapi/cross-platform/capture-10-images-to-ram/) | XiAPI (C) | Capture 10 frames into application-owned RAM and access them after closing the camera |
| [xiapiplus/cross-platform/capture-10-images](samples/xiapiplus/cross-platform/capture-10-images/) | xiAPIplus (C++) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-net-csharp/capture-10-images](samples/xiapi-net-csharp/capture-10-images/) | XiAPI.NET (C#) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-python/cross-platform/capture-10-images](samples/xiapi-python/cross-platform/capture-10-images/) | XiApiPython (Python) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi/hardware-specific/jetson-demosaic-sample](samples/xiapi/hardware-specific/jetson-demosaic-sample/) | XiAPI (C++/CUDA) | GPU demosaicing on Nvidia Jetson, bypassing xiAPI CPU processing |
| [xiapi-python/hardware-specific/gpio-samples-jetson](samples/xiapi-python/hardware-specific/gpio-samples-jetson/) | XiApiPython (Python) | GPIO, I2C, SPI and UART samples for the Jetson kit |

---

## Building samples

Build each sample separately by following the instructions in its own README.md file.
Repository-wide CI, build orchestration, and integration tests live in the
[Application-Samples-Infra](https://github.com/xiadmin/Application-Samples-Infra) repository,
which consumes this repository as a submodule. The tiny workflow under
`.github/workflows/` calls that reusable CI workflow with the exact pushed commit.

---

## Links

- [XIMEA API Manual](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA .NET API Manual](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Python API Manual](https://www.ximea.com/support/wiki/apis/XiAPI_Python_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
