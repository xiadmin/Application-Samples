# dependencies

All external libraries the samples use are located in this folder.

---

## Contents

| Path | What it is |
|---|---|
| `Dependencies.cmake` | C/C++ libraries: URL, SHA256 and build options of each archive. |
| `python-constraints.txt` | Python packages: pinned versions with the SHA-256 of every file PyPI publishes for them. |
| `Directory.Packages.props` | NuGet packages: central versions. |
| `Dependencies.props` | MSBuild logic for every C# sample, imported by the root `Directory.Build.props`. |
| `nuget.config` | NuGet settings for downloads: nuget.org only, its signature required. |
| `nuget-local.config` | NuGet settings for your own packages in `local/nuget/`: no remote source, no signature check. |
| `install-opencv-windows.ps1` | Windows only: downloads the pinned OpenCV installer, checks its SHA256 and unpacks it. |
| `local/` | Copies you provide yourself: `cmake/`, `nuget/` and `wheels/`; see [Providing a library yourself](#providing-a-library-yourself). Not committed. |
| `cache/` | What builds download automatically. |

---

## Providing a library yourself

Without `AUTOMATICALLY_DOWNLOAD_DEPENDENCIES`, a sample uses only copies you provide. They are used as they are, at any version.

### C / C++: `local/cmake/<library>/`

Build the library and install it into its own folder under `local/cmake/`. `find_package` searches every folder there, so the sample configures with no extra options. For example, libtiff from its source archive:

```bash
cmake -S <libtiff-source> -B <libtiff-build> -DCMAKE_INSTALL_PREFIX=dependencies/local/cmake/libtiff -DBUILD_SHARED_LIBS=OFF
cmake --build <libtiff-build> --config Release
cmake --install <libtiff-build> --config Release
```

A prebuilt package that ships a CMake config (for example `lib/cmake/<Library>/`) can be unpacked into `local/cmake/<library>/` as is. A library installed with your system package manager (apt, Homebrew, vcpkg) needs nothing here; point `CMAKE_PREFIX_PATH` at it only if `find_package` does not find it.

### C#: `local/nuget/`

Put the `.nupkg` files of the packages the sample needs into `local/nuget/`, at any version (the lowest one is used if there are several), then build the sample. NuGet restores only from this folder, without a signature or lock-file check. A `.nupkg` can come from your own build or from nuget.org:

```bash
curl -L -o dependencies/local/nuget/bitmiracle.libtiff.net.2.4.660.nupkg https://www.nuget.org/api/v2/package/BitMiracle.LibTiff.NET/2.4.660
```

The packages are unpacked into `cache/nuget-packages-local/`, apart from checked downloads. After replacing a package with another build of the same version, delete that folder so the new one is used.

### Python: `local/wheels/`

Put the wheel files into `local/wheels/` and install them by name; no network access is needed, and nothing is hash-checked. For example, for a sample that uses numpy and tifffile:

```bash
python -m pip download -d dependencies/local/wheels numpy tifffile    # once, on a machine with network access
python -m pip install --no-index --find-links dependencies/local/wheels numpy tifffile
```

Wheels are specific to the platform and Python version, so download them for the machine that runs the sample. Installing through a sample's `requirements.txt` also finds wheels here, but then applies the pinned hashes. You can also install the packages any other way, for example with your system package manager.
