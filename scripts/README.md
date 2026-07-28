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
| `--run` | - | Attempt to run executable outputs after all selected samples build/check successfully. |
| `--run-timeout-seconds` | positive integer | Timeout for each runtime attempt. Default: `90`. |
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

When `--run` is passed and no selected sample failed or had a missing dependency, the script attempts discovered runnable CMake outputs, one runnable .NET output per sample, and each checked Python `main.py`. Skipped samples do not prevent runtime attempts for successful samples. Every runtime attempt is limited by `--run-timeout-seconds` and reported as one of:

- `passed`: the program exited with code `0`
- `no_camera_or_failed`: the program exited with a nonzero code
- `timed_out`: the program did not exit before the configured timeout
- `could_not_start`: the program could not be launched

Runtime results are reported separately and do not change the command's exit code. Runtime attempts are not made if any selected sample failed to build/check or had a missing dependency.

## `new-sample.py`

`new-sample.py` creates a new sample scaffold under the lowercase `samples/` tree.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `--path` | relative path | New sample path relative to `samples/`, for example `xiapi/cross-platform/capture-50-images`. This is the preferred non-interactive path selector. |
| `--api` | API folder | API folder used by the legacy split form, for example `xiapi`, `xiapiplus`, `xiapi-net-csharp`, or `xiapi-python`. |
| `--group` | `cross-platform` or `hardware-specific` | Sample group used by the legacy split form for grouped APIs. Supply it for a fully non-interactive `--yes` run when the selected API requires a group and `--path` is not used. |
| `--sample` | folder name | Sample leaf folder used by the legacy split form. |
| `--lang` | `c`, `cpp`, `csharp`, or `python` | Language/template to scaffold. Required with `--yes` when the path does not imply a known language. |
| `--csv-path` | CSV path | Populate generated intro-comment and README metadata from this CSV. The path is forwarded to both generators. |
| `--yes` | - | Skip the final confirmation prompt. Missing path components may still be requested interactively; a language that cannot be inferred must be supplied with `--lang`. |

Path and name values must use lowercase kebab-case for new folder segments. Existing path segments may be reused when they are directories. Leading and trailing separators are normalized; the script rejects empty internal path segments, `.` / `..`, and Windows-invalid path characters.

Run the interactive prompt from the repository root:

```bash
python scripts/new-sample.py
```

Supported language templates are:

- `c`
- `cpp`
- `csharp`
- `python`

The script creates the language scaffold, then invokes both `generate-intro-comments.py` and `generate-readmes.py`. If CSV metadata is selected, the same CSV path is passed to both generators. By default, path-derived values are used where available and unavailable metadata is rendered as `TODO`. If `--csv-path` is supplied but no CSV row matches the new sample, both generators skip it and the scaffold remains created without a generated intro or `README.md`.

## `generate-readmes.py`

`generate-readmes.py` generates sample `README.md` files from `scripts/templates/sample-readme.md`. With no parameters, it finds every discovered sample directory and creates or rewrites its README using path-derived metadata and `TODO` for unavailable values. Pass `--csv-path` to populate metadata from a CSV; blank displayed metadata fields are rendered as `TODO` unless the script derives the value independently. An existing README is replaced completely.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `--directory-path` | sample directory | Process only this sample directory. If omitted, all discovered sample directories under `samples/` are processed. |
| `--csv-path` | CSV path | Populate generated README metadata from this CSV. Sample directories without a matching CSV row are skipped. |

Rewrite all discovered sample READMEs:

```bash
python scripts/generate-readmes.py
```

Rewrite one sample README with CSV metadata:

```bash
python scripts/generate-readmes.py --directory-path samples/xiapi/example --csv-path metadata.csv
```

## `generate-intro-comments.py`

`generate-intro-comments.py` writes generated intro comments/docstrings to supported sample source files. With no parameters, it finds all supported sample entry files, derives the sample name, API type, and OS platform from their paths, and uses `TODO` for unavailable values. Pass `--csv-path` to populate metadata from a CSV; blank displayed metadata fields are rendered as `TODO` unless the script derives the value independently. A recognized existing intro—a leading `/* ... */` block, legacy consecutive `//` lines, or Python module docstring—is removed before the generated intro is written.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-h`, `--help` | - | Show command help and exit. |
| `--csv-path` | CSV path | Populate generated intro metadata from this CSV. |
| `--file-path` | source file | Process only this supported sample entry file. If omitted, all supported entry files under `samples/` are processed. |

Write intros to all supported sample source files:

```bash
python scripts/generate-intro-comments.py
```

Populate one file from CSV metadata:

```bash
python scripts/generate-intro-comments.py --csv-path ximea-samples.csv --file-path samples/xiapi/example/main.c
```

Supported source entry files are:

- `main.c` (multiline `/* ... */` intro block)
- `main.cpp` (multiline `/* ... */` intro block)
- `Program.cs` (multiline `/* ... */` intro block)
- `main.py` (multiline module docstring, preserving a shebang or coding line)

An existing leading `/* ... */` block, legacy consecutive `//` lines, or Python module docstring is replaced completely. A legacy `//` intro extends through all consecutive leading `//` lines. Generated C, C++, and C# intros all use multiline `/* ... */` comments; Python uses a multiline module docstring. For Python, a leading shebang or encoding line is preserved. Content after the recognized intro is preserved. Missing files, unsupported filenames, and files outside `samples/` are skipped. When `--csv-path` is provided, files that cannot be matched to a CSV row are also skipped.

