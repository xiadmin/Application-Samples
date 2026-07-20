# Scripts

This directory contains repository maintenance helpers for the application samples.

User-facing command-line scripts are documented below. `common.py` is a shared helper module, and files under `scripts/templates/` are templates consumed by these scripts rather than standalone maintenance commands.

## `build.py`

`build.py` is the canonical build/check entry point for the repository. It discovers samples under the lowercase `samples/` tree and runs the appropriate local build or check for each selected sample.

Run from the repository root:

```bash
python3 scripts/build.py
```

By default, this selects every discovered sample.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `--all` | - | Build/check all discovered samples. This is also the default when no selector is provided. |
| `--sample` | sample name or path | Build/check one sample selected by generated sample name, repository-relative path, `samples/...` path, or absolute path. Can be passed multiple times. |
| `--type` | `cmake`, `dotnet`, or `python` | Restrict the run to one sample type. Can be passed multiple times. |
| `--skip-platform-specific` | - | Skip samples that are not under a `cross-platform/` folder. |
| `--tui` | - | Open the interactive terminal selector after applying type/sample filters. |
| `--clean` | - | Delete root `build/`, `.cmake-tmp/`, and `.dotnet-tmp/` before running. |
| `--keep-temp` | - | Keep root `.cmake-tmp/` and `.dotnet-tmp/` directories after the run. |
| `--configuration` | configuration name | Build configuration for CMake and .NET samples. Default: `Release`. |

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

## `new-sample.py`

`new-sample.py` creates a new sample scaffold under the lowercase `samples/` tree.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `--path` | relative path | New sample path relative to `samples/`, for example `xiapi/cross-platform/capture-50-images`. This is the preferred non-interactive path selector. |
| `--api` | API folder | API folder used by the legacy split form, for example `xiapi`, `xiapiplus`, `xiapi.net-c#`, or `xiapi-python`. |
| `--group` | `cross-platform` or `hardware-specific` | Sample group used by the legacy split form for grouped APIs. Required with `--yes` when the selected API requires a group and `--path` is not used. |
| `--sample` | folder name | Sample leaf folder used by the legacy split form. |
| `--lang` | `c`, `cpp`, `csharp`, or `python` | Language/template to scaffold. Required with `--yes` when the path does not imply a known language. |
| `--use-csv` | - | Populate generated README metadata from the samples CSV when `ximea-samples.csv` exists and a matching row is found. Without this flag, README metadata comes from template placeholders. |
| `--yes` | - | Create without the final confirmation prompt when all required values are supplied. |

Path and name values must use lowercase kebab-case for new folder segments. Existing path segments may be reused when they are directories. The script rejects empty path segments, `.` / `..`, and Windows-invalid path characters.

Run the interactive prompt from the repository root:

```bash
python3 scripts/new-sample.py
```

Create a scaffold non-interactively by passing the full path relative to `samples/`:

```bash
python3 scripts/new-sample.py --path xiapi/cross-platform/capture-50-images --lang c --yes
```

The path form is preferred because it matches the repository layout directly. The script also accepts the legacy split fields when needed:

```bash
python3 scripts/new-sample.py --api xiapi --group cross-platform --sample capture-50-images --lang c --yes
```

Supported language templates are:

- `c`
- `cpp`
- `csharp`
- `python`

Scaffold source files and README content are generated from files under `scripts/templates/`. By default, the generated README uses template placeholder metadata. Pass `--use-csv` to populate README metadata from the samples CSV when a matching row exists.

```bash
python3 scripts/new-sample.py --path xiapi/cross-platform/capture-50-images --lang c --yes --use-csv
```

## `generate-readmes.py`

`generate-readmes.py` regenerates sample `README.md` files from `scripts/templates/sample-readme.md`. By default it writes template placeholder metadata and does not read the samples CSV. Pass `--use-csv` to populate metadata from the CSV.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `sample_dirs` | one or more sample directories | Optional positional list of sample directories to process. Defaults to all discovered sample directories. |
| `--use-csv` | - | Populate README template metadata from the samples CSV. Without this flag, CSV is not read. |
| `--csv` | CSV path | Path to the samples CSV used only with `--use-csv`. Default: `ximea-samples.csv` at the repository root. |
| `--samples` | directory path | Samples root directory. Defaults to the repository `samples/` directory. |
| `--template` | Markdown template path | README template path. Default: `scripts/templates/sample-readme.md`. |
| `--write` | - | Write changed `README.md` files. Without this flag, the script is a dry run. |
| `--check` | - | Return exit code `1` if any README would change. Intended for CI/check mode. |
| `--sample-dir` | sample directory | Legacy repeated flag for selecting sample directories. Can be passed multiple times and can be combined with positional `sample_dirs`. |

Dry-run all discovered sample directories:

```bash
python3 scripts/generate-readmes.py
```

Write all README changes:

```bash
python3 scripts/generate-readmes.py --write
```

Process one or more sample directories by passing folder paths as command parameters. In `--help`, these are shown as `[sample_dirs ...]`:

```bash
python3 scripts/generate-readmes.py samples/xiapi/cross-platform/capture-10-images
python3 scripts/generate-readmes.py samples/xiapi/cross-platform/capture-10-images samples/xiapi-python/cross-platform/capture-10-images
```

The legacy repeated flag is still supported:

```bash
python3 scripts/generate-readmes.py --sample-dir samples/xiapi/cross-platform/capture-10-images
```

Use check mode in CI or before committing generated README changes:

```bash
python3 scripts/generate-readmes.py --check
```

Populate README metadata from the samples CSV only when explicitly requested:

```bash
python3 scripts/generate-readmes.py --use-csv --csv samples.csv --write
```

Use a different Markdown template with `--template`:

```bash
python3 scripts/generate-readmes.py --template scripts/templates/sample-readme.md
```

The script skips missing directories, directories without a supported sample source/project file, and directories outside `samples/`. In `--use-csv` mode, it also skips directories that cannot be matched to a CSV row.

## `generate-intro-comments.py`

`generate-intro-comments.py` adds generated intro comments/docstrings to supported sample source files. By default it writes template placeholder metadata and does not read the samples CSV. Pass `--use-csv` to populate metadata from the CSV.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `files` | one or more source files | Optional positional list of sample source files to process. Defaults to all supported sample entry files. |
| `--use-csv` | - | Populate intro metadata from the samples CSV. Without this flag, CSV is not read. |
| `--csv` | CSV path | Path to the samples CSV used only with `--use-csv`. Default: `ximea-samples.csv` at the repository root. |
| `--samples` | directory path | Samples root directory. Defaults to the repository `samples/` directory. |
| `--write` | - | Write missing intro comments/docstrings. Without this flag, the script is a dry run. |
| `--check` | - | Return exit code `1` if any file would change. Intended for CI/check mode. |
| `--file` | source file | Legacy repeated flag for selecting source files. Can be passed multiple times and can be combined with positional `files`. |

Dry-run all supported sample source files:

```bash
python3 scripts/generate-intro-comments.py
```

Write missing intro comments:

```bash
python3 scripts/generate-intro-comments.py --write
```

Process one or more source files by passing file paths as command parameters. In `--help`, these are shown as `[files ...]`:

```bash
python3 scripts/generate-intro-comments.py samples/xiapi/cross-platform/capture-10-images/main.c
python3 scripts/generate-intro-comments.py samples/xiapi/cross-platform/capture-10-images/main.c samples/xiapiplus/cross-platform/capture-10-images/main.cpp
```

The legacy repeated flag is still supported:

```bash
python3 scripts/generate-intro-comments.py --file samples/xiapi/cross-platform/capture-10-images/main.c
```

Use check mode in CI or before committing generated source-header changes:

```bash
python3 scripts/generate-intro-comments.py --check
```

Populate intro metadata from the samples CSV only when explicitly requested:

```bash
python3 scripts/generate-intro-comments.py --use-csv --csv samples.csv --write
```

Supported source entry files are:

- `main.c` (`/* ... */` intro block)
- `main.cpp` (`// ...` intro block)
- `Program.cs` (`// ...` intro block)
- `main.py` (module docstring, preserving a shebang or coding line)

Existing intro comments/docstrings are left unchanged. Missing files, unsupported filenames, and files outside `samples/` are skipped. In `--use-csv` mode, files that cannot be matched to a CSV row are also skipped.

### Troubleshooting

List available samples and verify selector spelling by passing an unknown sample name:

```bash
python3 scripts/build.py --sample does-not-exist
```

Show CLI help:

```bash
python3 scripts/build.py --help
```
