# Application-Samples

Code samples for XIMEA cameras, organized by feature area. Each sample is
self-contained, and includes its own README.md file.

---

## General prerequisites

- XIMEA Software Package installed.
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

---

## Building samples

You may use `bulild.ps1` to build all samples directly.
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
- External dependencies declared in `requirements.txt` inside the sample folder.
- Use `argparse` for user inputs; include `if __name__ == "__main__":`.
- Module and public function/class docstrings required; tiny helpers do not
  need boilerplate.

---

## .NET conventions

- 

---

## Links

- [XIMEA API Manual](https://www.ximea.com/support/wiki/apis/xiapi_manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
