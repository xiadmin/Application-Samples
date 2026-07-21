This directory contains repository maintenance helpers for the application samples.

- User-facing command-line scripts are documented below.
- `common.py` is a shared helper module.
- Files under `scripts/templates/` are templates consumed by these scripts.

## `build.py`

`build.py` is the canonical build/check entry point for the repository. It discovers samples under the lowercase `samples/` tree and runs the appropriate local build or check for each selected sample.

Run from the repository root:

```bash
python scripts/build.py
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `--all` | - | Build/check all discovered samples without opening the interactive selector. |
| `--sample` | sample name or path | Build/check one sample selected by generated sample name, repository-relative path, `samples/...` path, or absolute path. Can be passed multiple times. |
| `--type` | `cmake`, `dotnet`, or `python` | Restrict the run to one sample type. Can be passed multiple times. |
| `--skip-platform-specific` | - | Skip samples that are not under a `cross-platform/` folder. |
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
python scripts/new-sample.py
```

Supported language templates are:

- `c`
- `cpp`
- `csharp`
- `python`

Scaffold source files and README content are generated from files under `scripts/templates/`. By default, the generated README uses template placeholder metadata.

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
python scripts/generate-readmes.py
```

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
python scripts/generate-intro-comments.py
```

Supported source entry files are:

- `main.c` (`/* ... */` intro block)
- `main.cpp` (`// ...` intro block)
- `Program.cs` (`// ...` intro block)
- `main.py` (module docstring, preserving a shebang or coding line)

Existing intro comments/docstrings are left unchanged. Missing files, unsupported filenames, and files outside `samples/` are skipped. In `--use-csv` mode, files that cannot be matched to a CSV row are also skipped.

