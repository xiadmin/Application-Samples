# Dependency Inventory

**Abbreviations (defined once)**

| Abbreviation | Meaning |
|---|---|
| SP | XIMEA Software Package |
| CUDA | Compute Unified Device Architecture |
| NPP | NVIDIA Performance Primitives |
| SDK | Software Development Kit |
| SHA | Secure Hash Algorithm |
| CI | Continuous Integration |
| SPI | Serial Peripheral Interface |
| I2C | Inter-Integrated Circuit |

---

## 1. Dependency Rules

1. Add only dependencies listed in this file.
2. Submit a pull request to update this file before adding a new dependency to any project.
3. Pin every dependency to an exact approved version. Exception: Jetson system dependencies follow the JetPack platform version (see Section 8).
4. Do not duplicate a dependency across language ecosystems when one approved entry covers it.
5. Do not fetch dependencies from unapproved sources.
6. Record the license reference for every dependency.

---

## 2. Approved Dependency Table

| Name | Approved Version | Language(s) | Delivery Mechanism | Platform | Source | License |
|---|---|---|---|---|---|---|
| XIMEA SP | 4.33 | C, C++, C#, Python | Vendor installer | Windows, Linux, macOS (sample-specific limits apply) | https://www.ximea.com/software-downloads | Proprietary vendor package terms |
| libtiff | 4.7.1 | C | CMake FetchContent (SHA-256 `f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69`) | Cross-platform | https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz | https://gitlab.com/libtiff/libtiff/-/blob/v4.7.1/LICENSE.md |
| gpiod | 2.4.0 | Python | pip | Linux only | https://pypi.org/project/gpiod/ | https://git.kernel.org/pub/scm/libs/libgpiod/libgpiod.git/tree/COPYING |
| spidev | 3.8 | Python | pip | Linux with SPI device support | https://pypi.org/project/spidev/ | https://github.com/doceme/py-spidev/blob/master/LICENSE.md |
| pyserial | 3.5 | Python | pip | Cross-platform | https://pypi.org/project/pyserial/ | https://github.com/pyserial/pyserial/blob/master/LICENSE.txt |
| CUDA Toolkit + NVIDIA Performance Primitives (NPP) | JetPack platform version | C++ | Jetson system image | Jetson only | https://developer.nvidia.com/embedded/jetpack | https://docs.nvidia.com/cuda/eula/index.html |
| OpenCV with CUDA | JetPack platform version | C++ | Jetson system package | Jetson only | https://opencv.org/releases/ | https://github.com/opencv/opencv/blob/4.x/LICENSE |
| libi2c | Jetson platform package version | Python scripts / system helper | System package / `install_libi2c.sh` | Jetson / Linux only | https://git.kernel.org/pub/scm/utils/i2c-tools/i2c-tools.git/ | https://git.kernel.org/pub/scm/utils/i2c-tools/i2c-tools.git/tree/COPYING |

> **NuGet note:** Future NuGet packages are controlled by `dependencies/Directory.Packages.props`. Do not add a NuGet package to this table unless that file has been updated first.

---

## 3. Project-Owned Library Table

| Library Name | Path | Status |
|---|---|---|
| None | — | No project-owned vendored libraries exist in this repository |

> **Note:** The former `libs/tiff-writer` directory has been removed. TIFF output now uses the central libtiff dependency fetched by CMake (see Section 2 and Section 4). The `libs/tiff-writer` path is not an approved entry; any recreation of a `libs/` directory must be listed in this table first.

---

## 4. CMake Package Update Procedure

1. Identify the new approved version and its official archive URL.
2. Download the archive and compute the SHA-256 hash.
3. Update this file: change the approved version and SHA-256 in Section 2.
4. Set `URL` and `URL_HASH` in the `FetchContent_Declare` call in `dependencies/Dependencies.cmake`.
5. Build all affected samples and confirm zero errors.
6. Submit the pull request with the `dependencies/Dependencies.cmake` change and the updated `DEPENDENCIES.md`.

---

## 5. Python Package Update Procedure

1. Identify the new approved version on PyPI.
2. Update this file: change the approved version in Section 2.
3. Update `dependencies/python-constraints.txt` to pin the new version. Do not add version pins to sample-local requirements files; sample requirements stay unpinned and use the central constraint.
4. Run the affected sample on target hardware and confirm correct operation.
5. Submit the pull request with the `dependencies/python-constraints.txt` change and the updated `DEPENDENCIES.md`.

> **Rule:** Do not install the `ximea` Python package from pip. The XIMEA SP installer supplies the `ximea` Python package. Use only the installer-supplied copy.

---

## 6. NuGet Package Update Procedure

1. Identify the new approved version on NuGet.org.
2. Update `dependencies/Directory.Packages.props`: change the version attribute.
3. Ensure each affected `.csproj` imports `dependencies/Directory.Packages.props` with the correct relative path.
4. Update this file: add or update the package entry in Section 2.
5. Build all affected C# projects and confirm zero errors.
6. Submit the pull request with the `dependencies/Directory.Packages.props` change and the updated `DEPENDENCIES.md`.

---

## 7. XIMEA SP Update Procedure

1. Download the new SP installer from https://www.ximea.com/software-downloads.
2. Update this file: change the approved version in Section 2.
3. Update CI provisioning and any Docker/VM images to install the new SP version.
4. Build and test all C, C++, C#, and Python samples that use the SP.
5. Submit the pull request with the updated `DEPENDENCIES.md` and any CI/provisioning script changes.

**Version enforcement note:** The current CI provisioning scripts use moving vendor download URLs and cannot prove that they installed version 4.33. Use a versioned installer or a pinned CI image before enforcing the package version in CI. Do not infer the SP version from filenames or directory names.

**Runtime API query:** After a camera is open, `XI_PRM_API_VERSION` reports the API version. This is a runtime check only. It is not a package-level pre-build verification method.

**Verification methods by language:**

| Language | Verification Method |
|---|---|
| C / C++ | Compiler resolves headers and link libraries from `XIMEA_SP_PATH` (if set) or the default install root `/opt/XIMEA` on Linux. |
| C# | `HintPath` in the `.csproj` resolves managed assemblies from `XIMEA_SP_PATH`. |
| Python | The installer adds the `ximea` package to the Python path. Confirm with `import ximea` in the target environment. |
| Hardware (runtime) | Open a camera and query `XI_PRM_API_VERSION` to confirm the API version at runtime. |

---

## 8. Jetson System Dependency Exception

CUDA Toolkit + NPP, OpenCV with CUDA, and libi2c are supplied by the Jetson system image or a Jetson-specific install script. Exact version numbers follow the JetPack platform version in use. These dependencies cannot be independently versioned or fetched via CMake or pip. When updating the JetPack version, re-test all Jetson samples and record the new platform version in Section 2.

---

## 9. Cross-Language Native Library Rule

If two language systems must load the same native library file, stop the change. Add one shared build or package process before you add the dependency.

Apply this rule before introducing any new native library. Coordinate the build step in CMake or the installer before writing language-specific bindings or wrappers.
