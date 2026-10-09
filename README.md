# Application-Samples

Code samples for XIMEA cameras, showing how to control your camera using the XIMEA APIs. Each sample is self-contained and includes its own README.md file with detailed instructions.

---

## Prerequisites

- [XIMEA Software Package (SP) 4.33](https://www.ximea.com/software-downloads).
- Python 3.11 or newer for Python samples.
- CMake 3.16 or newer plus a C/C++ compiler for xiAPI C/C++ samples.
- OpenCV 4 for the `*-opencv` xiAPIplus samples.
- OpenCV and Pillow, installed with pip from each sample's `requirements.txt`, for the `*-opencv` and `*-pillow` Python samples.
- .NET SDK 8.0 or newer for xiAPI.NET samples (C# and VB.NET).

---

## Repository layout

```text
samples/
  xiapi/                    # C samples using xiAPI, plus xiAPI-based hardware-specific samples
  xiapiplus/                # C++ samples using xiAPIplus
  gentl/                    # C++ samples using GenTL, alone or alongside xiAPI
  xiapi-net-csharp/         # C# samples using xiAPI.NET
  xiapi-net-vb/             # VB.NET samples using xiAPI.NET
  xiapi-python/             # Python samples using xiAPI Python bindings
cmake/                      # shared sample CMake support files
dependencies/
  Dependencies.cmake       # central C/C++ library list
  Directory.Packages.props # central NuGet package version control
  python-constraints.txt   # pinned Python package versions
Directory.Build.props       # points MSBuild at the central NuGet version list
nuget.config                # NuGet package store and sources
.deps/                      # shared download cache, not committed
```

CI, build orchestration, and the dependency policy tests live in the separate
`Application-Samples-Infra` repository, which consumes this repository as its
`Application-Samples` submodule.

Each sample has a README.md explaining what it does and how to use it.

---

## Available samples

| Sample | API / Language | What it shows |
|--------|-----------------|---------------|
| [xiapi/cross-platform/capture-10-images](samples/xiapi/cross-platform/capture-10-images/) | XiAPI (C) | Capture 10 frames and print basic frame information |
| [xiapi/cross-platform/capture-10-images-to-file](samples/xiapi/cross-platform/capture-10-images-to-file/) | XiAPI (C) | Capture 10 stabilized frames and save them as TIFF files |
| [xiapi/cross-platform/capture-10-images-to-ram](samples/xiapi/cross-platform/capture-10-images-to-ram/) | XiAPI (C) | Capture 10 frames into application-owned RAM and access them after closing the camera |
| [xiapi/cross-platform/error-handling](samples/xiapi/cross-platform/error-handling/) | XiAPI (C) | Detect `XI_INVALID_HANDLE`, and recover from an `XI_TIMEOUT` |
| [xiapi/cross-platform/capture-10-images-to-dng](samples/xiapi/cross-platform/capture-10-images-to-dng/) | XiAPI (C) | Capture 10 raw frames and save them as DNG files with the SP's DNG Store library |
| [xiapi/cross-platform/offline-processing](samples/xiapi/cross-platform/offline-processing/) | XiAPI (C) | Capture RAW8 frames, close the camera, then convert them with xiAPI offline processing and save as TIFF |
| [xiapiplus/cross-platform/capture-10-images](samples/xiapiplus/cross-platform/capture-10-images/) | xiAPIplus (C++) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapiplus/cross-platform/capture-10-images-to-file](samples/xiapiplus/cross-platform/capture-10-images-to-file/) | xiAPIplus (C++) | Capture 10 stabilized frames and save them as TIFF files |
| [xiapiplus/cross-platform/capture-10-images-to-ram](samples/xiapiplus/cross-platform/capture-10-images-to-ram/) | xiAPIplus (C++) | Capture 10 frames into application-owned RAM and access them after closing the camera |
| [xiapiplus/cross-platform/error-handling](samples/xiapiplus/cross-platform/error-handling/) | xiAPIplus (C++) | Detect `XI_INVALID_HANDLE`, and recover from an `XI_TIMEOUT` |
| [xiapiplus/cross-platform/capture-10-images-to-dng](samples/xiapiplus/cross-platform/capture-10-images-to-dng/) | xiAPIplus (C++) | Capture 10 raw frames and save them as DNG files with the SP's DNG Store library |
| [xiapiplus/cross-platform/offline-processing](samples/xiapiplus/cross-platform/offline-processing/) | xiAPIplus (C++) | Capture RAW8 frames, close the camera, then convert them with xiAPI offline processing and save as TIFF |
| [xiapiplus/cross-platform/capture-25-images-in-parallel](samples/xiapiplus/cross-platform/capture-25-images-in-parallel/) | xiAPIplus (C++) | Open several cameras in parallel, trigger the slaves from the master's GPO1, capture bursts on per-camera threads |
| [xiapiplus/cross-platform/capture-50-images-and-display-opencv](samples/xiapiplus/cross-platform/capture-50-images-and-display-opencv/) | xiAPIplus + OpenCV (C++) | Capture 50 frames as `cv::Mat` and show them in an OpenCV window |
| [xiapiplus/cross-platform/capture-50-images-to-video-file-opencv](samples/xiapiplus/cross-platform/capture-50-images-to-video-file-opencv/) | xiAPIplus + OpenCV (C++) | Capture 50 grayscale frames, show them and record them to a Motion JPEG AVI file |
| [gentl/cross-platform/capture-10-images-to-file](samples/gentl/cross-platform/capture-10-images-to-file/) | GenTL (C++) | Load the GenTL producer (.cti) at run time, capture 10 frames through pure GenTL and save them as TIFF files |
| [gentl/cross-platform/capture-50-images-zero-copy](samples/gentl/cross-platform/capture-50-images-zero-copy/) | GenTL (C++) | Configure the camera with xiAPI, capture 50 frames through its GenTL data stream into application-allocated buffers |
| [gentl/cross-platform/capture-10-concatenated-images-to-file](samples/gentl/cross-platform/capture-10-concatenated-images-to-file/) | GenTL (C++) | Capture buffers of 5 concatenated images, split them and save every image as a TIFF file |
| [xiapi-net-csharp/capture-10-images](samples/xiapi-net-csharp/capture-10-images/) | XiAPI.NET (C#) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-net-csharp/capture-10-images-to-file](samples/xiapi-net-csharp/capture-10-images-to-file/) | XiAPI.NET (C#) | Capture 10 stabilized frames and save them as TIFF files |
| [xiapi-net-csharp/capture-10-images-to-ram](samples/xiapi-net-csharp/capture-10-images-to-ram/) | XiAPI.NET (C#) | Capture 10 frames into application-owned RAM and access them after closing the camera |
| [xiapi-net-csharp/error-handling](samples/xiapi-net-csharp/error-handling/) | XiAPI.NET (C#) | Detect `XI_INVALID_HANDLE`, and recover from an `XI_TIMEOUT` |
| [xiapi-net-csharp/capture-10-images-to-dng](samples/xiapi-net-csharp/capture-10-images-to-dng/) | XiAPI.NET (C#) | Capture 10 raw frames and save them as DNG files with the SP's DNG Store library |
| [xiapi-net-csharp/offline-processing](samples/xiapi-net-csharp/offline-processing/) | XiAPI.NET (C#) | Capture RAW8 frames, close the camera, then convert them with xiAPI offline processing and save as TIFF |
| [xiapi-net-csharp/display-images-in-window](samples/xiapi-net-csharp/display-images-in-window/) | XiAPI.NET (C#) | Live preview in a Windows Forms window with an exposure slider |
| [xiapi-net-vb/capture-100-images-to-file](samples/xiapi-net-vb/capture-100-images-to-file/) | XiAPI.NET (VB.NET) | Capture 100 grayscale frames and save them as JPEG files with WPF |
| [xiapi-python/cross-platform/capture-10-images](samples/xiapi-python/cross-platform/capture-10-images/) | XiApiPython (Python) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-python/cross-platform/capture-10-images-to-file](samples/xiapi-python/cross-platform/capture-10-images-to-file/) | XiApiPython (Python) | Capture 10 stabilized frames and save them as TIFF files |
| [xiapi-python/cross-platform/capture-10-images-to-ram](samples/xiapi-python/cross-platform/capture-10-images-to-ram/) | XiApiPython (Python) | Capture 10 frames into application-owned RAM and access them after closing the camera |
| [xiapi-python/cross-platform/error-handling](samples/xiapi-python/cross-platform/error-handling/) | XiApiPython (Python) | Detect `XI_INVALID_HANDLE`, and recover from an `XI_TIMEOUT` |
| [xiapi-python/cross-platform/capture-10-images-to-dng](samples/xiapi-python/cross-platform/capture-10-images-to-dng/) | XiApiPython (Python) | Capture 10 raw frames and save them as DNG files with the SP's DNG Store library |
| [xiapi-python/cross-platform/capture-from-two-cameras](samples/xiapi-python/cross-platform/capture-from-two-cameras/) | XiApiPython (Python) | Open two cameras, split the interface bandwidth between them and capture one frame from each |
| [xiapi-python/cross-platform/display-image-opencv](samples/xiapi-python/cross-platform/display-image-opencv/) | XiApiPython (Python) | Capture one MONO8, RGB24 or MONO16 frame and show it in an OpenCV window |
| [xiapi-python/cross-platform/display-video-opencv](samples/xiapi-python/cross-platform/display-video-opencv/) | XiApiPython (Python) | Live MONO8 or RGB24 video in an OpenCV window with an elapsed-time overlay |
| [xiapi-python/cross-platform/display-image-pillow](samples/xiapi-python/cross-platform/display-image-pillow/) | XiApiPython (Python) | Capture one MONO8, RGB24 or RGB32 frame and open it with Pillow in the system image viewer |
| [xiapi-python/cross-platform/save-image-pillow](samples/xiapi-python/cross-platform/save-image-pillow/) | XiApiPython (Python) | Capture one frame and save it with Pillow in the format given by the file extension |
| [xiapi/hardware-specific/jetson-demosaic-sample](samples/xiapi/hardware-specific/jetson-demosaic-sample/) | XiAPI (C++/CUDA) | GPU demosaicing on Nvidia Jetson, bypassing xiAPI CPU processing |
| [xiapi-python/hardware-specific/gpio-samples-jetson](samples/xiapi-python/hardware-specific/gpio-samples-jetson/) | XiApiPython (Python) | GPIO, I2C, SPI and UART samples for the Jetson kit |

LabVIEW examples are not part of this repository. They ship with the XIMEA Software Package when its optional "NI Labview interface - xiLib" component is installed, under `VisionLibraries\xiLib\xiLIB_64bit\APP_REQUIREMENTS\Examples` (or `xiLIB_32bit`) in the SP installation folder.

---

## Building samples

Build each sample separately by following the instructions in its own README.md file.

### Dependency cache and offline builds

An external package is downloaded once per machine and reused by every sample.
All three language tracks share one directory, `.deps/` at the repository root,
and none of them lets a sample pick its own version.

| Track | Shared store | Wired up by |
|-------|--------------|-------------|
| C / C++ | `.deps/downloads/` (archives) | `DOWNLOAD_DIR` in `application_samples_use()` |
| C# | `.deps/nuget-packages/` (extracted packages) | `globalPackagesFolder` in `nuget.config` |
| Python | `.deps/wheels/` (offline wheelhouse) | `--find-links` in each sample `requirements.txt` |

#### C / C++

A cached archive is reused as it is; its contents are not verified. A version
change in `dependencies/Dependencies.cmake` changes the URL and therefore the
archive name, so the new version is downloaded. If an upstream project ever
republishes a different archive under the same URL, delete the cached file to
pick it up. Warm the cache once with network access, then build offline:

```bash
cmake -B <build-dir> -DAPPLICATION_SAMPLES_OFFLINE=ON
```

With `APPLICATION_SAMPLES_OFFLINE=ON`, a missing archive fails immediately with
the URL to fetch by hand, instead of retrying the download for several minutes.

| Option | Meaning |
|--------|---------|
| `APPLICATION_SAMPLES_CACHE_DIR` | Where downloaded archives live. Defaults to `.deps/` at the repository root; point it at a shared drive or a CI cache to reuse it across checkouts. |
| `APPLICATION_SAMPLES_OFFLINE` | `ON` fails fast on a cache miss instead of reaching for the network. |

#### C\#

`nuget.config` redirects the package store into `.deps/nuget-packages/` and
clears machine-wide package sources, so a restore only uses the feeds listed
there. Versions come from `dependencies/Directory.Packages.props`, which the
repository-root `Directory.Build.props` feeds to MSBuild through
`DirectoryPackagesPropsPath`; a sample writes a `PackageReference` with no
`Version` and never imports that file itself.

Drop `.nupkg` files into `.deps/nuget-packages/` ahead of time, or restore once
with network access, then build offline:

```bash
dotnet build --no-restore
```

#### Python

`requirements.txt` points at the wheelhouse with `--find-links`, resolved
relative to the requirements file, and pip ignores it when the directory is
absent. Versions come from `dependencies/python-constraints.txt` via the `-c`
line in the same file.

Fill the wheelhouse on the machine that will run the sample, because wheels are
specific to platform and Python version:

```bash
python3 -m pip download -r requirements.txt -d <repo>/.deps/wheels   # once, online
python3 -m pip install --no-index -r requirements.txt                # offline
```

pip keeps its own download cache outside the repository. Point it at `.deps/`
too when a CI job caches only that directory:

```bash
export PIP_CACHE_DIR=<repo>/.deps/pip-cache
```

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
