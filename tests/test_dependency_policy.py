"""
Dependency policy tests.

Enforces the repository's dependency management policies:
  1. CMake: FetchContent_Declare must only appear in dependencies/Dependencies.cmake,
     not in any sample CMakeLists.txt under samples/.
  2. Python: gpiod, spidev, pyserial must not carry any version specifier in
     sample requirements files (only the central constraints file may pin them).
  3. C#: each project must import dependencies/Directory.Packages.props, and
     <PackageReference> must not carry a Version attribute or child element
     (namespace-aware; Version matched case-insensitively).
  4. libs/: must contain only project-owned entries listed in the explicit
     Markdown table under '## 3. Project-Owned Library Table' in DEPENDENCIES.md.
     Prose mentioning a former path does not authorise it.  If the table has no
     data rows (only a None/empty row), an absent libs/ directory passes and any
     recreated top-level libs/ entry fails.
  5. Central policy files must exist.
"""

import re
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


# ---------------------------------------------------------------------------
# 1. CMake policy
# ---------------------------------------------------------------------------

class TestCMakeDependencyPolicy(unittest.TestCase):
    """FetchContent_Declare must only appear in dependencies/Dependencies.cmake."""

    def test_dependencies_cmake_exists(self):
        deps_file = REPO_ROOT / "dependencies" / "Dependencies.cmake"
        self.assertTrue(
            deps_file.exists(),
            f"dependencies/Dependencies.cmake does not exist at {deps_file}",
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


# ---------------------------------------------------------------------------
# 2. Python policy
# ---------------------------------------------------------------------------

class TestPythonDependencyPolicy(unittest.TestCase):
    """Pinned/versioned packages gpiod, spidev, pyserial are forbidden in sample requirements files."""

    PINNED_PACKAGES = ["gpiod", "spidev", "pyserial"]

    # Any version specifier operator, or a direct URL/VCS reference
    _VERSION_PATTERN = re.compile(
        r"^[ \t]*("
        + "|".join(PINNED_PACKAGES)
        + r")(?:\s*\[[^]]+\])?\s*(?:===|==|>=|<=|~=|!=|>|<|@)",
        re.IGNORECASE,
    )

    def test_python_constraints_file_exists(self):
        constraints = REPO_ROOT / "dependencies" / "python-constraints.txt"
        self.assertTrue(
            constraints.exists(),
            f"dependencies/python-constraints.txt does not exist at {constraints}",
        )

    def _scan_requirements_file(self, path: Path) -> list[str]:
        """Return violating lines (package name only) from one requirements file."""
        hits = []
        for raw_line in path.read_text(errors="replace").splitlines():
            # Strip inline comments
            line = raw_line.split("#")[0]
            # Skip constraint/include directives (-c, -r, -f, --...) and blank lines
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                continue
            m = self._VERSION_PATTERN.match(line)
            if m:
                hits.append(m.group(1).lower())
        return hits

    def test_no_direct_version_specifiers_for_hardware_packages(self):
        """Sample requirements files must not version-pin gpiod, spidev, or pyserial."""
        central = (REPO_ROOT / "dependencies" / "python-constraints.txt").resolve()
        samples_root = REPO_ROOT / "samples"
        violations = []

        for req_file in samples_root.rglob("requirements*.txt"):
            if req_file.resolve() == central or _is_generated_path(req_file):
                continue
            hits = self._scan_requirements_file(req_file)
            if hits:
                violations.append(
                    f"{req_file.relative_to(REPO_ROOT)}: versioned {set(hits)}"
                )

        self.assertEqual(
            violations,
            [],
            "Direct version specifiers found for hardware packages in sample requirements:\n  "
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

    def test_no_packagereference_version_in_csproj(self):
        violations = []
        for csproj in REPO_ROOT.rglob("*.csproj"):
            if _is_generated_path(csproj):
                continue
            try:
                tree = ET.parse(csproj)
            except ET.ParseError as exc:
                self.fail(f"Could not parse {csproj}: {exc}")

            root = tree.getroot()
            rel = str(csproj.relative_to(REPO_ROOT))

            # Iterate all elements; match by local name to be namespace-agnostic
            for elem in root.iter():
                if _local_name(elem.tag) != "PackageReference":
                    continue
                pkg = elem.get("Include") or elem.get("Update") or "<unknown>"

                # Check Version attribute (case-insensitive key scan)
                for attr_key in elem.attrib:
                    if _local_name(attr_key).lower() == "version":
                        violations.append(
                            f"{rel}: PackageReference '{pkg}' has Version attribute"
                        )
                        break

                # Check <Version> child element (case-insensitive local name)
                for child in elem:
                    if _local_name(child.tag).lower() == "version":
                        violations.append(
                            f"{rel}: PackageReference '{pkg}' has <Version> child element"
                        )

        self.assertEqual(
            violations,
            [],
            "Versioned PackageReferences found (use dependencies/Directory.Packages.props):\n  "
            + "\n  ".join(violations),
        )


# ---------------------------------------------------------------------------
# 4. libs/ policy
# ---------------------------------------------------------------------------

class TestLibsDirectoryPolicy(unittest.TestCase):
    """libs/ must contain only project-owned entries listed in the explicit
    Markdown table under DEPENDENCIES.md §3.

    Parsing rules:
    - Only pipe-delimited table rows inside the section are examined.
    - The header row and separator row are skipped.
    - A row whose first data cell is 'None' (case-insensitive, ignoring
      whitespace and backticks) does not authorise any libs/ entry.
    - Prose text, blockquotes, and list items are ignored entirely.
    - Stops at the next '##' heading outside the section.
    """

    _SECTION_HEADER = re.compile(r"^##\s+3\.\s+Project-Owned Library Table", re.IGNORECASE)
    _NEXT_SECTION = re.compile(r"^##\s+")
    # Matches a Markdown table separator row: cells contain only dashes and spaces
    _SEPARATOR_ROW = re.compile(r"^\|[-| ]+\|$")

    def test_dependencies_md_exists(self):
        deps_md = REPO_ROOT / "DEPENDENCIES.md"
        self.assertTrue(
            deps_md.exists(),
            f"DEPENDENCIES.md does not exist at {deps_md}",
        )

    def _get_listed_libs(self) -> set[str]:
        """
        Parse only approved table rows in '## 3. Project-Owned Library Table'.

        Returns the set of authorised top-level libs/ names.  A table whose
        only data row has a first cell of 'None' returns an empty set.
        """
        deps_md = REPO_ROOT / "DEPENDENCIES.md"
        if not deps_md.exists():
            return set()

        lines = deps_md.read_text(errors="replace").splitlines()
        in_section = False
        header_seen = False
        entries: set[str] = set()

        for line in lines:
            if not in_section:
                if self._SECTION_HEADER.match(line):
                    in_section = True
                continue
            # Stop at the next section heading
            if self._NEXT_SECTION.match(line) and not self._SECTION_HEADER.match(line):
                break
            # Only process pipe-delimited table rows
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            # Skip separator rows (|---|---|)
            if self._SEPARATOR_ROW.match(stripped):
                continue
            # First table row is the header
            if not header_seen:
                header_seen = True
                continue
            # Data row: extract the first cell (Library Name column)
            cells = [c.strip().strip("`") for c in stripped.strip("|").split("|")]
            if not cells:
                continue
            name = cells[0].strip()
            # 'None' row means no approved libs
            if name.lower() == "none" or not name:
                continue
            # Authorise the top-level libs/<name> entry
            # Support explicit `libs/<name>/` paths in the cell too
            m = re.match(r"libs/([^/]+)", name)
            if m:
                entries.add(m.group(1))
            else:
                entries.add(name)

        return entries

    def test_libs_contains_only_listed_entries(self):
        libs_dir = REPO_ROOT / "libs"
        if not libs_dir.exists():
            # Absent libs/ is acceptable when DEPENDENCIES.md §3 table has no approved rows
            return
        listed = self._get_listed_libs()
        actual = {p.name for p in libs_dir.iterdir()}
        unlisted = actual - listed
        self.assertEqual(
            unlisted,
            set(),
            f"libs/ contains entries not listed in DEPENDENCIES.md §3 table: {unlisted}",
        )


if __name__ == "__main__":
    unittest.main()
