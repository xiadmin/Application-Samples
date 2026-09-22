# Application-Samples

Code samples for XIMEA cameras, showing how to control your camera using the XIMEA APIs. Each sample is self-contained and includes its own README.md file with detailed instructions.

---

## Prerequisites

- [XIMEA Software Package (SP) 4.33](https://www.ximea.com/software-downloads).
- Python 3.11 or newer for Python samples.
- CMake 3.16 or newer plus a C/C++ compiler for xiAPI C/C++ samples.
- .NET SDK 8.0 or newer for xiAPI.NET samples.

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
| [xiapiplus/cross-platform/capture-10-images](samples/xiapiplus/cross-platform/capture-10-images/) | xiAPIplus (C++) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-net-csharp/capture-10-images](samples/xiapi-net-csharp/capture-10-images/) | XiAPI.NET (C#) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi-python/cross-platform/capture-10-images](samples/xiapi-python/cross-platform/capture-10-images/) | XiApiPython (Python) | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [xiapi/hardware-specific/jetson-demosaic-sample](samples/xiapi/hardware-specific/jetson-demosaic-sample/) | XiAPI (C++/CUDA) | GPU demosaicing on Nvidia Jetson, bypassing xiAPI CPU processing |
| [xiapi-python/hardware-specific/gpio-samples-jetson](samples/xiapi-python/hardware-specific/gpio-samples-jetson/) | XiApiPython (Python) | GPIO, I2C, SPI and UART samples for the Jetson kit |

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
