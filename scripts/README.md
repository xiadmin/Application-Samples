# Scripts

This directory contains repository maintenance helpers for the application samples.

## `build.py`

`build.py` is the canonical build/check entry point for the repository. It discovers samples under the lowercase `samples/` tree and runs the appropriate local build or check for each selected sample.

Run from the repository root:

```bash
python3 scripts/build.py
```

By default, this selects every discovered sample.

### Discovered sample types

The script discovers samples by build entry file:

- C/C++ CMake samples: any `CMakeLists.txt` under `samples/`
- C# samples: any `.csproj` under `samples/`
- Python samples: any `main.py` under `samples/`

Sample names are derived from their path relative to `samples/` by joining path parts with `-`.

Example:

```text
samples/xiapi/cross-platform/capture-10-images
```

becomes:

```text
xiapi-cross-platform-capture-10-images
```

### What each sample type does

#### C/C++ CMake samples

Each CMake sample is configured and built independently through its own sample `CMakeLists.txt`:

```bash
cmake -S samples/<sample-path> -B .cmake-tmp/<sample-name>
cmake --build .cmake-tmp/<sample-name> --config Release
```

After a successful build, binary outputs are copied to:

```text
build/<sample-name>/
```

CMake samples require the CMake CLI, a working compiler/toolchain, and the XIMEA SDK expected by the sample CMake files.

#### C# samples

Each `.csproj` sample is built independently:

```bash
dotnet build samples/<sample-path>/<project>.csproj -c Release --output .dotnet-tmp/<sample-name>
```

Direct build outputs are copied to:

```text
build/<sample-name>/
```

C# samples require the `dotnet` CLI and the XIMEA SDK expected by the project.

#### Python samples

Python samples are checked with `py_compile` only:

```bash
python3 -m py_compile samples/<sample-path>/main.py
```

The script does not import or execute Python camera code, does not check for the `ximea` module.

### Selecting samples

Build/check all samples:

```bash
python3 scripts/build.py
python3 scripts/build.py --all
```

Select by sample name:

```bash
python3 scripts/build.py --sample xiapi-cross-platform-capture-10-images
```

Select by sample path:

```bash
python3 scripts/build.py --sample samples/xiapi/cross-platform/capture-10-images
```

Select multiple samples:

```bash
python3 scripts/build.py --sample xiapi-cross-platform-capture-10-images --sample xiapi-python-cross-platform-capture-10-images
```

Select by type:

```bash
python3 scripts/build.py --type cmake
python3 scripts/build.py --type dotnet
python3 scripts/build.py --type python
python3 scripts/build.py --type cmake --type python
```

### Platform-specific samples

Only samples under a `cross-platform/` folder are considered cross-platform. Samples without that folder, such as the current `xiapi.net-c#` layout, are treated as platform-specific for `--skip-platform-specific`.

Skip them with:

```bash
python3 scripts/build.py --skip-platform-specific
```

Skipped samples are still shown in the summary as `skipped`; they are not silently omitted.

### Interactive selector

Use the simple terminal selector:

```bash
python3 scripts/build.py --tui
```

The selector accepts:

- `a` or Enter for all listed samples
- `q` to quit without building
- numbers such as `1`
- comma-separated numbers such as `1,3,5`
- ranges such as `1-3`

### Cleanup behavior

Root temporary directories are removed after the run by default:

```text
.cmake-tmp/
.dotnet-tmp/
```

Keep them for debugging with:

```bash
python3 scripts/build.py --keep-temp
```

Use `--clean` to delete root build and temp directories before the run:

```bash
python3 scripts/build.py --clean
```

For each selected sample, these sample-local generated work directories are removed by default after the sample is processed:

```text
.cmake-tmp/
.dotnet-tmp/
__pycache__/
build/
bin/
obj/
```

The script also removes the selected sample's root output folder before rebuilding it:

```text
build/<sample-name>/
```

### Build configuration

CMake and .NET builds use `Release` by default.

Override it with:

```bash
python3 scripts/build.py --configuration Debug
```

### Result statuses

The summary reports one status per selected sample:

- `ok`: sample built or checked successfully
- `failed`: build/check command failed or expected outputs were not found
- `missing_dependency`: required build tool such as `cmake` or `dotnet` is unavailable
- `skipped`: sample was selected but skipped, for example by `--skip-platform-specific`

Exit code:

- `0`: no selected sample failed and no selected sample had missing dependencies
- `1`: at least one selected sample failed or had missing dependencies

Skipped samples do not make the command fail.

### CI usage

Python-only checks are suitable for a generic GitHub Actions runner:

```bash
python3 -m py_compile scripts/build.py scripts/common.py
python3 scripts/build.py --type python
```

CMake and .NET builds require CI jobs with the needed tools and XIMEA SDK configuration installed:

```bash
python3 scripts/build.py --type cmake --skip-platform-specific
python3 scripts/build.py --type dotnet
```

If those dependencies are not installed, the script reports `missing_dependency` and exits non-zero.

### Troubleshooting

List available samples and verify selector spelling by passing an unknown sample name:

```bash
python3 scripts/build.py --sample does-not-exist
```

Show CLI help:

```bash
python3 scripts/build.py --help
```
