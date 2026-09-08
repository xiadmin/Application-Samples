"""
Dependency policy tests.

Enforces the repository's dependency management policies:
  1. External package versions belong only in the applicable central file.
  2. Samples may request centrally versioned packages but may not override them.
  3. Project-owned source libraries inside a sample remain permitted.
  4. Exact dependency versions must not conflict across CMake, Python, and .NET
     source-owned build files.
  5. The three central build-system dependency files must exist.
"""

import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
GENERATED_DIRECTORY_NAMES = {".venv", "bin", "build", "obj"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_name(tag: str) -> str:
    """Strip XML namespace from a tag, e.g. '{ns}Foo' -> 'Foo'."""
    return tag.split("}")[-1] if "}" in tag else tag


def _is_generated_path(path: Path) -> bool:
    """Return True for repository-local generated or temporary trees."""
    return any(
        part in GENERATED_DIRECTORY_NAMES
        or part.startswith(".cmake-tmp")
        or part.startswith(".dotnet-tmp")
        for part in path.relative_to(REPO_ROOT).parts
    )


def _normalize_dependency_name(name: str) -> str:
    """Return a common key for dependency names from supported build systems."""
    return re.sub(r"[-_.]+", "-", name).casefold()


def _version_from_archive_url(url: str) -> str | None:
    """Get a dotted version from a dependency archive URL."""
    filename = url.rsplit("/", 1)[-1]
    matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.-]+)?)", filename)
    return matches[-1] if matches else None


def _cmake_source_text(path: Path) -> str:
    """Read CMake source without bracket or line comments."""
    text = path.read_text(errors="replace")
    text = re.sub(r"#\[\[.*?\]\]", "", text, flags=re.DOTALL)
    return re.sub(r"(?m)#.*$", "", text)


def _cmake_fetchcontent_declarations(path: Path) -> list[tuple[str, str | None]]:
    """Read each FetchContent name and its exact version, when present."""
    text = _cmake_source_text(path)
    declarations = []
    for match in re.finditer(
        r"FetchContent_Declare\s*\(\s*([A-Za-z0-9_.+-]+)(.*?)\)",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        name, body = match.groups()
        version = None
        tag = re.search(r"\bGIT_TAG\s+[\"']?([^\s\"')]+)", body, re.IGNORECASE)
        if tag:
            tag_value = tag.group(1)
            if re.fullmatch(
                r"[vV]?[0-9]+(?:\.[0-9]+)+(?:[-+][A-Za-z0-9.-]+)?",
                tag_value,
            ):
                version = tag_value.removeprefix("v").removeprefix("V")
        else:
            url = re.search(r"\bURL\s+[\"']?([^\s\"')]+)", body, re.IGNORECASE)
            if url:
                version = _version_from_archive_url(url.group(1))
        declarations.append((name, version))
    return declarations


def _cmake_dependency_versions(path: Path) -> list[tuple[str, str]]:
    """Read exact FetchContent and find_package versions."""
    declarations = [
        (name, version)
        for name, version in _cmake_fetchcontent_declarations(path)
        if version is not None
    ]

    for match in re.finditer(
        r"find_package\s*\(\s*([A-Za-z0-9_.+-]+)\s+"
        r"([0-9]+(?:\.[0-9]+)+(?:[-+][A-Za-z0-9.-]+)?)(?![0-9.])",
        _cmake_source_text(path),
        re.IGNORECASE,
    ):
        declarations.append(match.groups())
    return declarations


def _cmake_versioned_find_packages(path: Path) -> list[tuple[str, str]]:
    """Read every sample find_package call that selects a version or range."""
    unversioned_options = {
        "BYPASS_PROVIDER",
        "COMPONENTS",
        "CONFIG",
        "CONFIGS",
        "EXACT",
        "GLOBAL",
        "HINTS",
        "MODULE",
        "NAMES",
        "NO_MODULE",
        "NO_POLICY_SCOPE",
        "OPTIONAL_COMPONENTS",
        "PATHS",
        "QUIET",
        "REGISTRY_VIEW",
        "REQUIRED",
        "UNWIND_INCLUDE",
    }
    declarations = []
    for match in re.finditer(
        r"find_package\s*\(\s*([A-Za-z0-9_.+-]+)(?:\s+([^\s)]+))?",
        _cmake_source_text(path),
        re.IGNORECASE,
    ):
        name, second_argument = match.groups()
        if second_argument and second_argument.upper() not in unversioned_options:
            declarations.append((name, second_argument))
    return declarations


def _python_dependency_versions(path: Path) -> list[tuple[str, str]]:
    """Read exact pins from a Python requirements or constraints file."""
    declarations = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(
            r"([A-Za-z0-9_.-]+)(?:\s*\[[^]]+\])?\s*(?:===|==)\s*([^\s;]+)",
            line,
        )
        if match:
            declarations.append(match.groups())
    return declarations


def _is_python_sample_local_source(path: Path, line: str) -> bool:
    """Return True for a source path that stays below the requirements folder."""
    candidate = None
    if line.startswith("-e ") or line.startswith("--editable "):
        candidate = line.split(maxsplit=1)[1]
    elif " @ file:" in line:
        candidate = line.split(" @ file:", 1)[1]
    elif line.startswith("file:"):
        candidate = line.removeprefix("file:")
    elif line.startswith(("./", ".\\", "../", "..\\")):
        candidate = line

    if not candidate or "://" in candidate:
        return False
    source = (path.parent / candidate).resolve()
    return source.is_relative_to(path.parent.resolve())


def _python_requirement_entries(path: Path) -> list[tuple[str, bool]]:
    """Read package names and whether each sample entry selects a version."""
    declarations = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if _is_python_sample_local_source(path, line):
            continue
        if line.startswith("-e ") or line.startswith("--editable "):
            declarations.append((line, True))
            continue
        if line.startswith("-"):
            continue
        match = re.match(
            r"([A-Za-z0-9_.-]+)(?:\s*\[[^]]+\])?\s*(.*)$",
            line,
            re.IGNORECASE,
        )
        if not match:
            continue
        name, remainder = match.groups()
        has_version = bool(
            re.match(r"(?:===|==|>=|<=|~=|!=|>|<|@)", remainder.strip())
        )
        declarations.append((name, has_version))
    return declarations


def _python_constraint_references(path: Path) -> set[Path]:
    """Resolve constraint files activated by one requirements file."""
    references = set()
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        value = None
        if line.startswith("-c ") or line.startswith("--constraint "):
            value = line.split(maxsplit=1)[1]
        elif line.startswith("--constraint="):
            value = line.split("=", 1)[1]
        if value:
            references.add((path.parent / value).resolve())
    return references


def _dotnet_dependency_versions(path: Path) -> list[tuple[str, str]]:
    """Read package versions and overrides from an MSBuild file."""
    root = ET.parse(path).getroot()
    declarations = []
    for element in root.iter():
        if _local_name(element.tag) not in {"PackageVersion", "PackageReference"}:
            continue
        name = element.get("Include") or element.get("Update")
        version = next(
            (
                value
                for key, value in element.attrib.items()
                if _local_name(key).casefold() in {"version", "versionoverride"}
            ),
            None,
        )
        if version is None:
            version = next(
                (
                    (child.text or "").strip()
                    for child in element
                    if _local_name(child.tag).casefold()
                    in {"version", "versionoverride"}
                ),
                None,
            )
        if name and version:
            declarations.append((name, version))
    return declarations


def _dotnet_sample_build_files() -> list[Path]:
    """Return source-owned MSBuild files that can declare sample dependencies."""
    samples = REPO_ROOT / "samples"
    files = {
        path
        for pattern in ("*.csproj", "*.props", "*.targets")
        for path in samples.rglob(pattern)
        if not _is_generated_path(path)
    }
    return sorted(files)


def _collect_dependency_versions() -> dict[str, list[tuple[str, str]]]:
    """Collect exact dependency versions from source-owned build files."""
    declarations: dict[str, list[tuple[str, str]]] = {}

    dependency_files = [REPO_ROOT / "dependencies" / "Dependencies.cmake"]
    dependency_files.extend((REPO_ROOT / "samples").rglob("CMakeLists.txt"))
    for path in dependency_files:
        if path.exists() and not _is_generated_path(path):
            for name, version in _cmake_dependency_versions(path):
                declarations.setdefault(_normalize_dependency_name(name), []).append(
                    (version, str(path.relative_to(REPO_ROOT)))
                )

    python_files = [REPO_ROOT / "dependencies" / "python-constraints.txt"]
    python_files.extend((REPO_ROOT / "samples").rglob("requirements*.txt"))
    for path in python_files:
        if path.exists() and not _is_generated_path(path):
            for name, version in _python_dependency_versions(path):
                declarations.setdefault(_normalize_dependency_name(name), []).append(
                    (version, str(path.relative_to(REPO_ROOT)))
                )

    dotnet_files = [REPO_ROOT / "dependencies" / "Directory.Packages.props"]
    dotnet_files.extend(_dotnet_sample_build_files())
    for path in dotnet_files:
        if path.exists() and not _is_generated_path(path):
            for name, version in _dotnet_dependency_versions(path):
                declarations.setdefault(_normalize_dependency_name(name), []).append(
                    (version, str(path.relative_to(REPO_ROOT)))
                )

    return declarations


class TestDependencyVersionParsers(unittest.TestCase):
    """Keep each supported build-file parser covered by permanent fixtures."""

    def _file(self, name: str, content: str) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / name
        path.write_text(content)
        return path

    def test_cmake_versions(self):
        path = self._file(
            "Dependencies.cmake",
            """
FetchContent_Declare(foo URL https://example.test/foo-1.2.3.tar.gz)
FetchContent_Declare(baz GIT_TAG v3.4.5)
FetchContent_Declare(unversioned GIT_TAG main)
find_package(bar 2.3.4 EXACT REQUIRED)
find_package(ranged 1.0...<2.0 REQUIRED)
find_package(variable ${PACKAGE_VERSION} REQUIRED)
find_package(system REQUIRED)
# find_package(ignored 9.9.9 REQUIRED)
""",
        )
        self.assertEqual(
            _cmake_dependency_versions(path),
            [("foo", "1.2.3"), ("baz", "3.4.5"), ("bar", "2.3.4")],
        )
        self.assertEqual(
            _cmake_versioned_find_packages(path),
            [
                ("bar", "2.3.4"),
                ("ranged", "1.0...<2.0"),
                ("variable", "${PACKAGE_VERSION}"),
            ],
        )

    def test_python_versions(self):
        path = self._file(
            "requirements.txt",
            "Foo_Bar[extra]==1.2.3 ; python_version >= '3.11'\n",
        )
        self.assertEqual(_python_dependency_versions(path), [("Foo_Bar", "1.2.3")])

    def test_python_sample_entries_and_constraint_reference(self):
        path = self._file(
            "requirements.txt",
            "-c constraints.txt\nfoo\nbar>=2\n"
            "-e ./local-helper\nlocal @ file:./local-package\n"
            "-e git+https://example.test/baz.git\n",
        )
        self.assertEqual(
            _python_requirement_entries(path),
            [("foo", False), ("bar", True), ("-e git+https://example.test/baz.git", True)],
        )
        self.assertEqual(
            _python_constraint_references(path),
            {(path.parent / "constraints.txt").resolve()},
        )

    def test_dotnet_versions_and_overrides(self):
        path = self._file(
            "Directory.Packages.props",
            """<Project xmlns="urn:test"><ItemGroup>
  <PackageVersion Include="foo" Version="1.2.3" />
  <PackageReference Include="bar"><VersionOverride>2.3.4</VersionOverride></PackageReference>
</ItemGroup></Project>""",
        )
        self.assertEqual(
            _dotnet_dependency_versions(path),
            [("foo", "1.2.3"), ("bar", "2.3.4")],
        )

    def test_project_owned_source_libraries_are_not_package_versions(self):
        cmake = self._file(
            "CMakeLists.txt",
            "add_library(sample_helper STATIC helper.c)\n",
        )
        project = self._file(
            "Sample.csproj",
            """<Project><ItemGroup>
  <ProjectReference Include="../SampleHelper/SampleHelper.csproj" />
</ItemGroup></Project>""",
        )

        self.assertEqual(_cmake_dependency_versions(cmake), [])
        self.assertEqual(_dotnet_dependency_versions(project), [])


# ---------------------------------------------------------------------------
# 1. CMake policy
# ---------------------------------------------------------------------------

class TestCMakeDependencyPolicy(unittest.TestCase):
    """External CMake dependency versions stay in Dependencies.cmake."""

    def test_dependencies_cmake_exists(self):
        deps_file = REPO_ROOT / "dependencies" / "Dependencies.cmake"
        self.assertTrue(
            deps_file.exists(),
            f"dependencies/Dependencies.cmake does not exist at {deps_file}",
        )

    def test_central_fetchcontent_dependencies_have_exact_versions(self):
        """Every fetched external dependency has a versioned URL or tag."""
        deps_file = REPO_ROOT / "dependencies" / "Dependencies.cmake"
        text = _cmake_source_text(deps_file)
        declared = {
            _normalize_dependency_name(name)
            for name in re.findall(
                r"FetchContent_Declare\s*\(\s*([A-Za-z0-9_.+-]+)",
                text,
                re.IGNORECASE,
            )
        }
        versioned = {
            _normalize_dependency_name(name)
            for name, _ in _cmake_dependency_versions(deps_file)
        }

        self.assertEqual(
            declared - versioned,
            set(),
            "FetchContent dependencies without an exact version in "
            f"dependencies/Dependencies.cmake: {declared - versioned}",
        )

    def test_fetchcontent_only_in_dependencies_cmake(self):
        """No sample CMakeLists.txt under samples/ may contain FetchContent_Declare."""
        samples_root = REPO_ROOT / "samples"
        violations = []

        # Only scan CMakeLists.txt files that live under samples/,
        # excluding any .cmake-tmp build trees.
        for cmake_file in samples_root.rglob("CMakeLists.txt"):
            if _is_generated_path(cmake_file):
                continue
            text = cmake_file.read_text(errors="replace")
            if re.search(r"FetchContent_Declare", text, re.IGNORECASE):
                violations.append(str(cmake_file.relative_to(REPO_ROOT)))

        self.assertEqual(
            violations,
            [],
            "FetchContent_Declare found outside dependencies/Dependencies.cmake:\n  "
            + "\n  ".join(violations),
        )

    def test_versioned_find_package_only_in_dependencies_cmake(self):
        """A sample can find a package, but it cannot select its version."""
        violations = []
        for cmake_file in (REPO_ROOT / "samples").rglob("CMakeLists.txt"):
            if _is_generated_path(cmake_file):
                continue
            for name, version in _cmake_versioned_find_packages(cmake_file):
                violations.append(
                    f"{cmake_file.relative_to(REPO_ROOT)}: {name} {version}"
                )

        self.assertEqual(
            violations,
            [],
            "Sample-local external package versions found; declare them in "
            "dependencies/Dependencies.cmake:\n  "
            + "\n  ".join(violations),
        )


# ---------------------------------------------------------------------------
# 2. Python policy
# ---------------------------------------------------------------------------

class TestPythonDependencyPolicy(unittest.TestCase):
    """Python requirements use only exact versions from the central constraints."""

    def test_python_constraints_file_exists(self):
        constraints = REPO_ROOT / "dependencies" / "python-constraints.txt"
        self.assertTrue(
            constraints.exists(),
            f"dependencies/python-constraints.txt does not exist at {constraints}",
        )

    def test_central_constraints_use_exact_versions(self):
        """Each central Python dependency has one exact version."""
        constraints = REPO_ROOT / "dependencies" / "python-constraints.txt"
        exact_names = {
            _normalize_dependency_name(name)
            for name, _ in _python_dependency_versions(constraints)
        }
        violations = [
            name
            for name, _ in _python_requirement_entries(constraints)
            if _normalize_dependency_name(name) not in exact_names
        ]

        self.assertEqual(
            violations,
            [],
            "Python constraints without an exact == or === version:\n  "
            + "\n  ".join(violations),
        )

    def test_sample_requirements_use_central_versions(self):
        """Every requested package is central, versionless, and constrained."""
        central = (REPO_ROOT / "dependencies" / "python-constraints.txt").resolve()
        central_names = {
            _normalize_dependency_name(name)
            for name, _ in _python_dependency_versions(central)
        }
        samples_root = REPO_ROOT / "samples"
        violations = []

        for req_file in samples_root.rglob("requirements*.txt"):
            if req_file.resolve() == central or _is_generated_path(req_file):
                continue
            entries = _python_requirement_entries(req_file)
            if entries and central not in _python_constraint_references(req_file):
                violations.append(
                    f"{req_file.relative_to(REPO_ROOT)}: does not activate "
                    "dependencies/python-constraints.txt"
                )
            for name, has_version in entries:
                normalized = _normalize_dependency_name(name)
                if has_version:
                    violations.append(
                        f"{req_file.relative_to(REPO_ROOT)}: {name} selects a local version"
                    )
                if normalized not in central_names:
                    violations.append(
                        f"{req_file.relative_to(REPO_ROOT)}: {name} has no central version"
                    )

        self.assertEqual(
            violations,
            [],
            "Python dependency policy violations:\n  "
            + "\n  ".join(violations),
        )


# ---------------------------------------------------------------------------
# 3. C# policy
# ---------------------------------------------------------------------------

class TestCSharpDependencyPolicy(unittest.TestCase):
    """C# projects import central package control and keep references versionless."""

    def test_directory_packages_props_exists(self):
        props = REPO_ROOT / "dependencies" / "Directory.Packages.props"
        self.assertTrue(
            props.exists(),
            f"dependencies/Directory.Packages.props does not exist at {props}",
        )

    def test_central_package_versions_are_complete(self):
        """Every central PackageVersion has a package name and a version."""
        props = REPO_ROOT / "dependencies" / "Directory.Packages.props"
        root = ET.parse(props).getroot()
        violations = []
        for elem in root.iter():
            if _local_name(elem.tag) != "PackageVersion":
                continue
            package = elem.get("Include") or elem.get("Update")
            versions = [
                value
                for key, value in elem.attrib.items()
                if _local_name(key).casefold() == "version"
            ]
            versions.extend(
                (child.text or "").strip()
                for child in elem
                if _local_name(child.tag).casefold() == "version"
            )
            if not package or not any(versions):
                violations.append(ET.tostring(elem, encoding="unicode").strip())

        self.assertEqual(
            violations,
            [],
            "Incomplete PackageVersion entries in "
            "dependencies/Directory.Packages.props:\n  "
            + "\n  ".join(violations),
        )

    def test_csproj_imports_directory_packages_props(self):
        expected = (REPO_ROOT / "dependencies" / "Directory.Packages.props").resolve()
        violations = []

        for csproj in REPO_ROOT.rglob("*.csproj"):
            if _is_generated_path(csproj):
                continue
            try:
                root = ET.parse(csproj).getroot()
            except ET.ParseError as exc:
                self.fail(f"Could not parse {csproj}: {exc}")

            imports = [
                elem.get("Project", "")
                for elem in root.iter()
                if _local_name(elem.tag) == "Import"
            ]
            has_central_import = any(
                (csproj.parent / project.replace("\\", "/")).resolve() == expected
                for project in imports
                if project
            )
            if not has_central_import:
                violations.append(str(csproj.relative_to(REPO_ROOT)))

        self.assertEqual(
            violations,
            [],
            "C# projects missing dependencies/Directory.Packages.props import:\n  "
            + "\n  ".join(violations),
        )

    def test_sample_msbuild_files_do_not_declare_package_versions(self):
        """PackageVersion, Version, and VersionOverride belong in central props."""
        violations = []
        for build_file in _dotnet_sample_build_files():
            try:
                tree = ET.parse(build_file)
            except ET.ParseError as exc:
                self.fail(f"Could not parse {build_file}: {exc}")

            root = tree.getroot()
            rel = str(build_file.relative_to(REPO_ROOT))

            # Iterate all elements; match by local name to be namespace-agnostic
            for elem in root.iter():
                element_name = _local_name(elem.tag)
                if element_name == "PackageVersion":
                    pkg = elem.get("Include") or elem.get("Update") or "<unknown>"
                    violations.append(f"{rel}: local PackageVersion '{pkg}'")
                    continue
                if element_name != "PackageReference":
                    continue
                pkg = elem.get("Include") or elem.get("Update") or "<unknown>"

                # Check Version and VersionOverride attributes.
                for attr_key in elem.attrib:
                    metadata = _local_name(attr_key).casefold()
                    if metadata in {"version", "versionoverride"}:
                        violations.append(
                            f"{rel}: PackageReference '{pkg}' has {metadata} attribute"
                        )
                        break

                # Check <Version> and <VersionOverride> child elements.
                for child in elem:
                    metadata = _local_name(child.tag).casefold()
                    if metadata in {"version", "versionoverride"}:
                        violations.append(
                            f"{rel}: PackageReference '{pkg}' has <{metadata}> child element"
                        )

        self.assertEqual(
            violations,
            [],
            "Sample-local NuGet versions found; use "
            "dependencies/Directory.Packages.props:\n  "
            + "\n  ".join(violations),
        )

    def test_package_references_have_central_versions(self):
        """Each external NuGet package has one version in central props."""
        props = REPO_ROOT / "dependencies" / "Directory.Packages.props"
        root = ET.parse(props).getroot()
        central_names = {
            _normalize_dependency_name(package)
            for elem in root.iter()
            if _local_name(elem.tag) == "PackageVersion"
            for package in [elem.get("Include") or elem.get("Update")]
            if package
        }
        violations = []

        for build_file in _dotnet_sample_build_files():
            try:
                root = ET.parse(build_file).getroot()
            except ET.ParseError as exc:
                self.fail(f"Could not parse {build_file}: {exc}")
            for elem in root.iter():
                if _local_name(elem.tag) != "PackageReference":
                    continue
                package = elem.get("Include") or elem.get("Update")
                if package and _normalize_dependency_name(package) not in central_names:
                    violations.append(
                        f"{build_file.relative_to(REPO_ROOT)}: {package} has no central version"
                    )

        self.assertEqual(
            violations,
            [],
            "NuGet packages missing from dependencies/Directory.Packages.props:\n  "
            + "\n  ".join(violations),
        )


# ---------------------------------------------------------------------------
# 4. Cross-file version policy
# ---------------------------------------------------------------------------

class TestCrossFileDependencyVersions(unittest.TestCase):
    """A dependency can occur in many build files only at one exact version."""

    def test_no_dependency_has_conflicting_exact_versions(self):
        conflicts = []
        for name, declarations in sorted(_collect_dependency_versions().items()):
            versions = {version for version, _ in declarations}
            if len(versions) > 1:
                details = ", ".join(
                    f"{version} in {path}" for version, path in declarations
                )
                conflicts.append(f"{name}: {details}")

        self.assertEqual(
            conflicts,
            [],
            "Dependencies have conflicting exact versions:\n  "
            + "\n  ".join(conflicts),
        )


if __name__ == "__main__":
    unittest.main()
