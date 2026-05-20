# Application-Samples

Code samples for XIMEA cameras, organized by feature area. Each sample is
self-contained, has a single clear learning goal, and includes its own README.

---

## Repository layout

    samples/
      <feature>/
        <concrete-topic>/
          <language>/         <- one sample implementation lives here
            README.md
            CMakeLists.txt    (C/C++ samples)
            main.c / main.cpp
            ...
    cmake/                    <- shared CMake modules (FindXIMEA, SampleDefaults)
    build/                    <- build output (generated, not committed)

New folder names use kebab-case. New C/C++/Python source files use snake_case.
New C#/VB.NET source files use PascalCase.

---

## Available samples

| Sample | Language | What it shows |
|--------|----------|---------------|
| [acquisition/capture-10-images/c](samples/acquisition/capture-10-images/c/) | C | Basic xiAPI acquisition: open camera, set exposure, grab 10 frames |

---

## Prerequisites (all samples)

- XIMEA Software Package installed. Set `XIMEA_SP_PATH` to its root directory
  (`C:\XIMEA` on Windows, `/opt/XIMEA` on Linux).
- CMake 3.16+ for C/C++ samples.
- Python 3.9+ for Python samples.

Each sample README lists its own exact requirements.

---

## Building C/C++ samples

The shared `cmake/FindXIMEA.cmake` module is prepended to `CMAKE_MODULE_PATH`
by each sample's `CMakeLists.txt`, so samples build standalone without a
top-level super-build.

    # Linux
    cd samples/acquisition/capture-10-images/c
    cmake -B .cmake-tmp -DXIMEA_SP_PATH=/opt/XIMEA
    cmake --build .cmake-tmp

    # Windows (PowerShell)
    cd samples\acquisition\capture-10-images\c
    cmake -B .cmake-tmp -A x64 -DXIMEA_SP_PATH=C:\XIMEA
    cmake --build .cmake-tmp

Output goes to `build/<sample-name>/`. On Windows, runtime DLLs land in
`build/_dependencies/`. The `.cmake-tmp/` work directory can be deleted after
a successful build.

---

## C/C++ conventions

These rules apply to all C and C++ samples in this repository. AGENTS.md is
the authoritative source; the rules below are reproduced here for quick
reference.

### Language standard

- C samples target C11.
- C++ samples target C++17 or newer.

### Naming

| Construct | Style | Example |
|-----------|-------|---------|
| Functions | lowerCamelCase | `runCapture`, `getFrameCount` |
| Classes / structs | UpperCamelCase (PascalCase) | `FrameBuffer`, `CameraConfig` |
| Macros | SCREAMING_SNAKE_CASE | `MAX_FRAME_COUNT`, `XI_CHECK` |
| Constants (const variables) | lowerCamelCase — NOT caps | `frameCount`, `exposureUs` |
| Local variables | lowerCamelCase or snake_case, descriptive | `grabTimeoutMs`, `frame_index` |
| Member variables | lowerCamelCase with `m_` prefix | `m_width`, `m_frameBuffer` |
| Global variables | avoid; if unavoidable, prefix `g_` | `g_instance` |
| Namespaces | lowercase | `ximea`, `utils` |
| Enum types | PascalCase | `PixelFormat`, `TriggerMode` |
| Enum values | ALL_CAPS or PascalCase | `PIXEL_FORMAT_RAW8` / `RisingEdge` |
| Template parameters | Single uppercase letter or PascalCase | `T`, `KeyType` |
| Pointer variables | prefix `p_` or suffix `Ptr` (optional but consistent) | `p_node`, `dataPtr` |
| Boolean variables | prefix `is`, `has`, `can`, `should` | `isValid`, `hasData` |
| File names | snake_case | `main.c`, `frame_buffer.cpp` |

Note: `const` variables are NOT all-caps — only macros use SCREAMING_SNAKE_CASE.
Single-character names are acceptable for loop indices (`i`, `j`, `k`) only.

### File and project structure

- `main.c` / `main.cpp` lives at the sample implementation root only — never
  inside `include/` or `src/`.
- Reusable declarations go in `include/`, implementation files in `src/`.
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
- Doxygen-style docblocks are required for reusable public APIs and helper
  libraries; skip boilerplate on trivial local helpers in demo code.

---

## Python conventions

- Target Python 3.9+.
- External dependencies declared in `requirements.txt` inside the sample folder.
- Use `argparse` for user inputs; include `if __name__ == "__main__":`.
- Module and public function/class docstrings required; tiny helpers do not
  need boilerplate.

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
5. Follow the naming and structure rules above.

---

## Links

- [XIMEA API Manual](https://www.ximea.com/support/wiki/allprod/XIMEA_API_Manual)
- [XIMEA Linux Software Package](https://www.ximea.com/support/wiki/allprod/XIMEA_Linux_Software_Package)
- [XIMEA Windows Software Package](https://www.ximea.com/support/wiki/allprod/XIMEA_Windows_Software_Package)
