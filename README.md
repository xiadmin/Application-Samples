# Application-Samples

Code samples for XIMEA cameras, organized by feature area. Each sample is
self-contained, and includes its own README.md file.

---

## General prerequisites

- XIMEA Software Package installed (latest beta).
- CMake 3.16+.

---

## Repository layout

    samples/
      <feature>/
        <concrete-topic>/
          <language>/         <- one sample implementation lives here
            README.md
            CMakeLists.txt    
            main.c / main.cpp / main.py / main.cs
            ...
    cmake/                    <- shared CMake modules
    build/                    <- build output 

---

## Available samples

| Sample | Language | What it shows |
|--------|----------|---------------|
| [acquisition/capture-10-images/c](samples/acquisition/capture-10-images/c/) | C | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [acquisition/capture-10-images/cpp](samples/acquisition/capture-10-images/cpp/) | C++ | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [acquisition/capture-10-images/csharp](samples/acquisition/capture-10-images/csharp/) | C# | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |
| [acquisition/capture-10-images/python](samples/acquisition/capture-10-images/python/) | Python | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |

---

## Building samples

You may use `build.ps1` to build all samples directly.
Alternatively, you can build each sample separately by following the instructions in its own README.md file.

---

## Adding a new sample

1. Create `samples/<feature>/<concrete>/<language>/`.
2. Write a `README.md` that includes: one-line purpose, prerequisites, build
   steps, run command, expected output, known limitations, and links to XIMEA
   docs.
3. For C/C++: add `CMakeLists.txt` using `find_package(XIMEA REQUIRED)` and
   prepend `cmake/` to `CMAKE_MODULE_PATH`.
4. Do not hardcode camera serial numbers, IP addresses, or machine-specific
   paths — use CLI arguments, environment variables, or a config file.
5. Follow the naming and structure rules below.

---

## Languages standards

- C11.
- C++17 or newer.
- Python 3.9 or newer.
- .NET 8 or newer.
---

## Naming convention

- Folders and executables names use kebab-case.
- New C/C++/Python source files use snake_case.
- New C#/VB.NET/Cmake source files use PascalCase.

---

## C/C++ conventions

### Naming

| Construct | Style | Example |
|-----------|-------|---------|
| Functions | lowerCamelCase | `runCapture`, `getFrameCount` |
| Classes / structs | UpperCamelCase (PascalCase) | `FrameBuffer`, `CameraConfig` |
| Macros | SCREAMING_SNAKE_CASE | `MAX_FRAME_COUNT`, `XI_CHECK` |
| Constants (const variables) | lowerCamelCase — NOT caps | `frameCount`, `exposureUs` |
| Local variables | lowerCamelCase | `grabTimeoutMs` |
| Member variables | lowerCamelCase with `m_` prefix | `m_width`, `m_frameBuffer` |
| Global variables | avoid; if unavoidable, prefix `g_` | `g_instance` |
| Namespaces | lowercase | `ximea`, `utils` |
| Enum types | PascalCase | `PixelFormat`, `TriggerMode` |
| Enum values | SCREAMING_SNAKE_CASE | `PIXEL_FORMAT_RAW8` |
| Template parameters | Single uppercase letter or PascalCase | `T`, `KeyType` |
| Pointer variables | prefix `p_` | `p_node` |
| Boolean variables | prefix `is`, `has`, `can`, `should` | `isValid`, `hasData` |
| File names | snake_case | `main.c`, `frame_buffer.cpp` |

Note: `const` variables are NOT all-caps — only macros use SCREAMING_SNAKE_CASE.
Single-character names are acceptable for loop indices (`i`, `j`, `k`) only.

### File and project structure

- Reusable declarations go in `include/`, implementation files in `src/`.
- `main.c` / `main.cpp` lives at the sample implementation root only — never
  inside subfolders.
- Every sample that produces a binary includes `CMakeLists.txt`.
- An optional `Makefile` may wrap CMake for convenience but must not replace it.

### CMake

- Use `target_*` commands — no broad global configuration.
- Use `find_package(...)` for dependencies when a Find module or config package
  is available (e.g. `find_package(XIMEA REQUIRED)`).
- Do not hardcode include directories or library paths.
- Shared modules live in `cmake/` and are prepended to `CMAKE_MODULE_PATH` by
  each sample; do not create a top-level monolithic CMakeLists.

### Code quality

- RAII and standard ownership patterns; avoid raw `new`/`delete` unless
  necessary.
- Do not use `using namespace std;` in headers.
- Write errors to `stderr` and return meaningful exit codes (`EXIT_SUCCESS` /
  `EXIT_FAILURE`).
- Do not use `goto`.
- Avoid Windows-specific types (`DWORD`, etc.) in cross-platform code.
- Comments explain non-obvious reasoning, hardware quirks, and API caveats —
  not obvious code.

---

## Python conventions

- Target Python 3.9+.
- The `ximea` package is installed by the XIMEA SDK into `site-packages/ximea` — no pip install needed.
- External runtime dependencies beyond the SDK are declared in `requirements.txt` inside the sample folder.
- Use `argparse` for user inputs when the sample accepts parameters; include `if __name__ == "__main__":`.
- Module and public function/class docstrings required; tiny helpers do not
  need boilerplate.

### Naming

| Construct | Style | Example |
|-----------|-------|---------|
| Functions | snake_case | `run_capture`, `get_frame_count` |
| Classes | UpperCamelCase (PascalCase) | `FrameBuffer`, `CameraConfig` |
| Constants (module-level) | SCREAMING_SNAKE_CASE | `MAX_FRAME_COUNT`, `DEFAULT_TIMEOUT_MS` |
| Local variables | snake_case | `grab_timeout_ms`, `frame_count` |
| Parameters | snake_case | `exposure_us`, `serial_number` |
| Instance attributes | snake_case | `self.width`, `self.frame_buffer` |
| Private attributes / methods | snake_case with `_` prefix | `_width`, `_validate_params` |
| Boolean variables | prefix `is_`, `has_`, `can_`, `should_` | `is_valid`, `has_data` |
| Type aliases | UpperCamelCase | `FrameList`, `ConfigDict` |
| Enum types | UpperCamelCase | `PixelFormat`, `TriggerMode` |
| Enum values | SCREAMING_SNAKE_CASE | `PIXEL_FORMAT_RAW8` |
| File names | snake_case | `main.py`, `frame_buffer.py` |

Note: module-level `Final` constants use SCREAMING_SNAKE_CASE; all other names do not.
Single-character names are acceptable for loop indices (`i`, `j`, `k`) only.

### File and project structure

- `main.py` is the entry point at the sample implementation root.
- Reusable helpers may be extracted into sibling modules (e.g. `camera_utils.py`)
  inside the same sample folder; do not copy helpers across samples.
- Every sample with pip-installable external dependencies includes `requirements.txt`.

### Code quality

- Use context managers (`with` statements) for resource cleanup — camera
  handles, file handles, etc.
- Write errors to `sys.stderr` (or via `argparse` error handling) and exit
  with a non-zero code on failure; do not silently swallow exceptions.
- Prefer f-strings over `%`-formatting or `.format()`.
- Do not use bare `except:` clauses; catch specific exception types.
- Comments explain non-obvious reasoning, hardware quirks, and API caveats —
  not obvious code.

---

## .NET conventions

- Target the latest LTS .NET version by default; add legacy targets only when there is a real compatibility requirement.
- Prefer one multi-target project over duplicated per-version folders when the code is materially the same.
- Catch exceptions at sensible boundaries and print actionable errors to `Console.Error`; do not wrap every single API call in a one-line `try/catch`.

### Naming

| Construct | Style | Example |
|-----------|-------|---------|
| Namespaces | PascalCase | `Ximea.Samples.Acquisition` |
| Classes / structs / records | PascalCase | `FrameBuffer`, `CameraConfig` |
| Interfaces | PascalCase with `I` prefix | `IFrameSource`, `ICameraHandle` |
| Methods | PascalCase | `RunCapture`, `GetFrameCount` |
| Properties | PascalCase | `ExposureUs`, `FrameCount` |
| Events | PascalCase | `FrameArrived`, `CameraDisconnected` |
| Constants (`const` / `static readonly`) | PascalCase | `MaxFrameCount`, `DefaultTimeoutMs` |
| Enum types | PascalCase | `PixelFormat`, `TriggerMode` |
| Enum values | PascalCase | `PixelFormatRaw8`, `TriggerModeSoftware` |
| Local variables | camelCase | `grabTimeoutMs`, `frameCount` |
| Parameters | camelCase | `exposureUs`, `serialNumber` |
| Private fields | camelCase with `_` prefix | `_width`, `_frameBuffer` |
| Boolean members | prefix `Is`, `Has`, `Can`, `Should` | `IsValid`, `HasData` |
| File names | PascalCase | `Main.cs`, `FrameBuffer.cs` |

Note: Microsoft naming guidelines do not use SCREAMING_SNAKE_CASE for any C# construct.
Single-character names are acceptable for loop indices (`i`, `j`, `k`) and generic type parameters (`T`, `TKey`) only.

### File and project structure

- One class (or closely related types) per file; file name matches the primary type name.
- `Program.cs` is the entry point at the sample implementation root.
- Every sample includes a `.csproj` file.
- Shared utility code lives in a separate project or folder — do not copy it across samples.

### Code quality

- Use `using` declarations / `IDisposable` patterns for resource cleanup; avoid manual `try/finally` dispose chains where `using` suffices.
- Prefer `async`/`await` over blocking `.Result` / `.Wait()` calls.
- Prefer `var` when the type is obvious from the right-hand side; use explicit types when it aids clarity.
- Use C# 8+ nullable reference types (`#nullable enable`) in new code.
- Do not use `goto`.
- Write XML doc comments (`/// <summary>`) for public types and members; tiny private helpers do not need boilerplate documentation.



---

## Links

- [XIMEA API Manual](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
