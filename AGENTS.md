## Repository layout

Use the API/language-first sample layout. The samples and the CI tooling are two
repositories; CI checks the samples out under `Application-Samples/`:

```text
Application-Samples/          sample repository
  samples/                  # sample source
  cmake/                    # shared sample CMake support files
  dependencies/             # the three central dependency files
Application-Samples-Infra/    CI repository
  scripts/                  # CI, build, and maintenance helpers
  tests/                    # integration and dependency policy tests
```

Layout rules:

- The canonical sample root is `samples/` in the Application-Samples repository.
- README links and generated paths must use lowercase `samples/`.
- New folder names should be kebab-case.
- Do not introduce spaces in new paths.
- Preserve existing sample-root names unless an explicit migration task says otherwise.
- `build/` may be used for local generated build output, but generated build artifacts must not be committed.
- Platform-specific samples (e.g. Jetson) may have their own hardware-specific constraints; do not put them to cross-platform samples folder.

## Build orchestration

Each buildable C or C++ sample should be buildable on its own and from the shared CMake entry point.

Rules:

- Each buildable C/C++ sample must include its own `CMakeLists.txt`.
- The shared CMake entry lives at `cmake/CMakeLists.txt`.
- Shared CMake helpers belong in `cmake/`.
- CMake logic must handle the lowercase `samples/` path correctly.
- Build outputs should not be committed.

C/C++ CMake requirements:

- Use `cmake_minimum_required(VERSION 3.16)` or newer.
- Use target-based CMake commands such as `target_link_libraries`, `target_include_directories`, and `target_compile_features`.
- Use `find_package` or shared CMake modules for dependencies; do not hard-code SDK include/library paths in sample CMake files.
- Include `install(TARGETS ...)` for executable targets.
- Add platform guards or platform-specific compile definitions where relevant.
- Platform-only samples must explicitly guard unsupported platforms and explain the required build option or platform.

## C and C++ conventions

Keep both C and C++ sample tracks where they are useful.

Language level:

- New C++ samples should use C++17.
- C samples should use C11 unless a sample has a specific reason not to.

Naming:

| Item | Convention |
| --- | --- |
| Functions | `lowerCamelCase` |
| Classes and structs | `UpperCamelCase` / PascalCase |
| Macros | `SCREAMING_SNAKE_CASE` |
| `const` variables | `lowerCamelCase`, not all-caps |
| Local variables | `lowerCamelCase` |
| Member variables | `m_` prefix, such as `m_width` |
| Global variables | Avoid; if unavoidable, use `g_` prefix |
| Namespaces | lowercase |
| Enum types | PascalCase |
| Enum values | `SCREAMING_SNAKE_CASE` |
| Template parameters | A single uppercase letter or PascalCase |
| Pointer variables | `p_` prefix |
| Boolean variables | Prefix with `is`, `has`, `can`, or `should` |
| File names | `snake_case` |

Style and structure:

- Single-character names are acceptable for loop indices (`i`, `j`, `k`) only.
- Include order for C++ files:
  1. system headers
  2. third-party headers
  3. XIMEA headers
  4. local headers
- Use RAII in C++ where possible.
- Avoid raw `new`/`delete` in C++ samples.
- Do not use `using namespace std;`.
- Do not use `goto`.
- Return `EXIT_SUCCESS` / `EXIT_FAILURE` or clear equivalent status codes.
- Print errors to stderr.
- Check every XIMEA API call that can fail with the `CE` macro from a nearby C sample, or with `try`/`catch` in C++ samples.
- Do not silently swallow errors in sample code.
- Minimal samples may keep a flat layout with `main.c` / `main.cpp` at the sample root.
- Larger multi-file samples should put reusable declarations in `include/` and implementation files in `src/`; keep `main.c` / `main.cpp` at the sample root.

## Python conventions

Repository Python code targets Python 3.11+.

General rules:

- Keep beginner samples simple; command-line arguments are optional for minimal samples.
- Use `argparse` when a sample needs configurable inputs such as camera index, exposure, frame count, output path, serial number, or IP address.
- Python samples do not need `requirements.txt` when they only depend on the XIMEA SDK-installed `ximea` package.
- Add `requirements.txt` only when the sample has pip-installable third-party dependencies.
- Scripts should be directly runnable with an `if __name__ == "__main__":` entry point.

Naming:

| Item | Convention |
| --- | --- |
| Functions | `snake_case` |
| Classes | `UpperCamelCase` / PascalCase |
| Module-level constants, including `Final` constants | `SCREAMING_SNAKE_CASE` |
| Local variables and parameters | `snake_case` |
| Instance attributes | `snake_case` |
| Private attributes and methods | `_snake_case` |
| Boolean variables | Prefix with `is_`, `has_`, `can_`, or `should_` |
| Type aliases and enum types | `UpperCamelCase` / PascalCase |
| Enum values | `SCREAMING_SNAKE_CASE` |
| File names | `snake_case` |

Style:

- Single-character names are acceptable for loop indices (`i`, `j`, `k`) only.
- Prefer context managers for camera handles, file handles, and other resources.
- Prefer f-strings over `%`-formatting or `.format()`.
- Do not use bare `except:` clauses; catch specific exception types.

## C# / xiAPI.NET conventions

General rules:

- Keep SDK DLL `HintPath` references as the canonical approach for XIMEA xiAPI.NET samples.
- Use latest LTS .NET by default for new C# samples.
- Use multi-targeting only when replacing or consolidating known legacy sample variants.
- Catch exceptions at sensible boundaries and print actionable errors with `Console.Error.WriteLine`; do not wrap every single API call in a one-line `try`/`catch` unless there is a sample-specific reason.
- Keep project files explicit and easy to build with `dotnet build`.
- Prefer one multi-target project over duplicated per-version folders when the code is materially the same.

Naming:

| Item | Convention |
| --- | --- |
| Namespaces, classes, structs, records, methods, properties, events, enum types, and enum values | PascalCase |
| Interfaces | PascalCase with `I` prefix |
| Constants (`const` / `static readonly`) | PascalCase |
| Local variables and parameters | camelCase |
| Private fields | `_camelCase` |
| Boolean members | Prefix with `Is`, `Has`, `Can`, or `Should` |
| File names | PascalCase |

Style and structure:

- Microsoft naming guidelines do not use `SCREAMING_SNAKE_CASE` for C# constructs.
- Single-character names are acceptable for loop indices (`i`, `j`, `k`) and generic type parameters (`T`, `TKey`) only.
- Use one class or closely related type group per file; the file name should match the primary type name.
- `Program.cs` is the entry point at the sample implementation root.
- Every C# sample includes a `.csproj` file.
- Shared utility code should live in a separate project or folder; do not copy it across samples.
- Use `using` declarations / `IDisposable` patterns for resource cleanup when appropriate.
- Prefer `async`/`await` over blocking `.Result` / `.Wait()` calls.
- Prefer `var` when the type is obvious from the right-hand side; use explicit types when it aids clarity.
- Use C# 8+ nullable reference types (`#nullable enable`) in new code.
- Do not use `goto`.

## Sample documentation

Every sample folder should have a `README.md`. Existing section names are acceptable if they clearly cover the required information.

A good sample README includes:

- one-line purpose/summary
- prerequisites: OS, hardware, SDK version, language runtime, and libraries
- build instructions when applicable
- run instructions
- expected output or behavior
- known limitations/caveats/TODOs
- links to relevant XIMEA documentation

Documentation rules:

- Commands should be copy-pasteable.
- Use lowercase `samples/` in paths.
- Do not document generated build artifacts as source files.
- Keep links current when files are moved.
- Follow the template in `scripts/templates/sample-readme.md`.

## Source comments and generated metadata

- Require generated intro comments/docstrings for samples where the repository generator supports them.
- Keep generated intro comments/docstrings concise and regenerate them through the repository scripts instead of hand-maintaining divergent boilerplate.
- Keep source comments useful and concise.
- Comment non-obvious logic, especially hardware/platform constraints and error-handling paths.
- Avoid comments that merely repeat the next line of code.

## Dependencies and configuration

Central library management: every external package version lives in exactly one of three central files, and a sample asks for a package without choosing its version. The requested package is then downloaded and built for that sample.

| Language | Central file | Sample asks for a package by |
| --- | --- | --- |
| C / C++ | `dependencies/Dependencies.cmake` | calling `application_samples_use(<name>)`, or an unversioned `find_package` |
| Python | `dependencies/python-constraints.txt` | activating it with `-c` and listing the bare package name |
| C# | `dependencies/Directory.Packages.props` | importing it and using a versionless `PackageReference` |

Rules:

- Require every external package version in one of the three central files.
- Do not permit a sample to override a central version: no `FetchContent_Declare`, `application_samples_dependency`, or versioned `find_package` in a sample `CMakeLists.txt`, no version specifier in a sample `requirements*.txt`, and no `Version` or `VersionOverride` on a sample `PackageReference`.
- Record a new C/C++ library as an `application_samples_dependency()` entry in `dependencies/Dependencies.cmake`. Give `NAME`, `URL`, `SHA256`, the `TARGET` the library defines, and the `OPTIONS` that configure its build. Write no new helper function; `application_samples_use(<name>)` reads the record.
- Permit project-owned source libraries inside a sample; use `add_library`, a local Python import, or `ProjectReference` for them. These are not external packages and carry no central version.
- Do not commit external source or built libraries; a dependency is fetched and built at configure time.
- One library must not appear at two different exact versions across the three central files.
- `tests/test_dependency_policy.py` in the `Application-Samples-Infra` repository enforces these rules in CI; run it there against this checkout before changing a dependency.
- Do not hard-code camera serial numbers, credentials, IP addresses, or user-specific paths in source.
- Prefer command-line arguments or documented configuration files for values that differ between users.
- Never commit secrets or machine-local configuration.
- XIMEA Software Package latest beta is the default SDK prerequisite unless a sample documents a more specific requirement.
- Use environment variables documented by the XIMEA SDK, such as `XIMEA_SP_PATH`, where appropriate.

## Scripts and generated files

- Keep maintenance scripts in `scripts/`.
- `scripts/new-sample.py` is the canonical new-sample scaffolding helper.
- Scripts should support the lowercase `samples/` tree.
- When editing generation scripts, verify them against a small temporary fixture or a focused dry run.