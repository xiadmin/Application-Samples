#!/usr/bin/env python3
"""Generate sample README.md files from a Markdown template and samples CSV."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from common import (
    CSV_NAME,
    clean,
    find_samples_root,
    folder_name_for,
    normalize,
    normalize_language,
    powershell_path,
    read_sample_rows,
    repo_root,
    split_list,
)

DEFAULT_TEMPLATE = "scripts/templates/sample-readme.md"
README_NAME_RE = re.compile(r"^readme\.md$", re.IGNORECASE)

SUPPORTED_SOURCE_FILES = {
    "main.c": "C",
    "main.cpp": "C++",
    "Program.cs": "C#",
    "main.py": "Python",
}

LINKS_BY_LANGUAGE = {
    "C": ["[xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)"],
    "C++": ["[xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)"],
    "C#": [
        "[xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)",
        "[XIMEA Software Packages](https://www.ximea.com/software-downloads)",
    ],
    "Python": [
        "[xiAPI Python documentation](https://www.ximea.com/support/wiki/apis/Python)",
        "[xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)",
    ],
}


@dataclass(frozen=True)
class SampleInfo:
    name: str
    author: str
    category: str
    description: str
    duplicate_group_id: str
    os_platform: str
    hardware_platform: str
    api_type: str
    language: str
    libraries: str
    year: str
    location: str
    path: str
    copyright: str
    note: str
    done: str


def display(value: str, default: str = "-") -> str:
    value = clean(value)
    return value if value else default


def markdown_escape(value: str) -> str:
    return display(value).replace("|", "\\|")


def read_samples(csv_path: Path) -> list[SampleInfo]:
    return [
        SampleInfo(
            name=data.get("Sample name", ""),
            author=data.get("Author", ""),
            category=data.get("Tomas_category", ""),
            description=data.get("Description", ""),
            duplicate_group_id=data.get("Duplicate_group_ID", ""),
            os_platform=data.get("OS platform", ""),
            hardware_platform=data.get("Hardware platform", ""),
            api_type=data.get("API type", ""),
            language=data.get("Programming language", ""),
            libraries=data.get("Libraries", ""),
            year=data.get("Year of last update", ""),
            location=data.get("Location", ""),
            path=data.get("Path", ""),
            copyright=data.get("Copyright", ""),
            note=data.get("Note", ""),
            done=data.get("Done?", ""),
        )
        for data in read_sample_rows(csv_path)
    ]


def language_for_dir(sample_dir: Path) -> str | None:
    for filename, language in SUPPORTED_SOURCE_FILES.items():
        if (sample_dir / filename).is_file():
            return language
    csproj_files = sorted(sample_dir.glob("*.csproj"))
    if csproj_files:
        return "C#"
    return None


def iter_sample_dirs(samples_root: Path) -> list[Path]:
    found: set[Path] = set()
    for filename in SUPPORTED_SOURCE_FILES:
        found.update(path.parent for path in samples_root.rglob(filename))
    found.update(path.parent for path in samples_root.rglob("*.csproj"))
    return sorted(found)


def readme_path(sample_dir: Path) -> Path:
    for item in sample_dir.iterdir():
        if item.is_file() and README_NAME_RE.match(item.name):
            return item
    return sample_dir / "README.md"


def path_score(row: SampleInfo, sample_dir: Path, samples_root: Path, language: str) -> int:
    rel = sample_dir.relative_to(samples_root)
    rel_norm = normalize(str(rel))
    folder_tokens = {normalize(part) for part in rel.parts if normalize(part)}
    sample_name = normalize(row.name)

    if not sample_name:
        return -1

    score = -1
    # Exact sample-name matches should beat fuzzy matches even when the CSV row's
    # language is less specific than the repository folder (for example the CSV
    # has a C++ row for Capture-10-images, while this repo also has C/Python
    # variants of that same sample).
    if sample_name in folder_tokens:
        score = 160
    elif sample_name and sample_name in rel_norm:
        score = 150
    else:
        words = [normalize(part) for part in re.split(r"[^A-Za-z0-9]+", row.name) if normalize(part)]
        matched_words = sum(1 for word in words if word and word in rel_norm)
        if words and matched_words == len(words):
            score = 90
        elif words and matched_words >= max(1, len(words) - 1):
            score = 60

    if score < 0:
        return -1

    row_lang = normalize_language(row.language)
    dir_lang = normalize_language(language)
    if row_lang == dir_lang:
        score += 50
    elif row_lang and dir_lang and row_lang != dir_lang:
        score -= 30

    api = normalize(row.api_type)
    if api and api in folder_tokens:
        score += 30
    hardware = normalize(row.hardware_platform)
    if hardware and hardware in rel_norm:
        score += 5
    os_platform = normalize(row.os_platform)
    if os_platform and os_platform in rel_norm:
        score += 3
    if clean(row.done).lower() in {"x", "yes", "true", "done"}:
        score += 2
    return score


def find_sample_info(sample_dir: Path, samples_root: Path, samples: list[SampleInfo], language: str) -> SampleInfo | None:
    scored = [(path_score(row, sample_dir, samples_root, language), row) for row in samples]
    scored = [(score, row) for score, row in scored if score >= 0]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def title_for(info: SampleInfo) -> str:
    return display(info.name)


def sdk_requirement(info: SampleInfo) -> str:
    if normalize(info.api_type) in {"none", ""}:
        return "Not required unless the sample accesses a XIMEA camera"
    return "4.32+"


def extra_prerequisites(info: SampleInfo, language: str) -> str:
    rows: list[tuple[str, str]] = []
    norm_lang = normalize_language(language)
    if norm_lang in {"c", "cpp"}:
        compiler = "MSVC 2022+, GCC 9+, or Clang 10+"
        if norm_lang == "cpp":
            compiler += " (C++17)"
        rows.extend([("CMake", "3.16 or newer"), ("Compiler", compiler)])
    elif norm_lang == "csharp":
        rows.append((".NET SDK", "8.0 or newer"))
    elif norm_lang == "python":
        rows.append(("Python", "3.9+"))

    for library in split_list(info.libraries):
        rows.append(("Library", library))

    if not rows:
        return ""
    return "".join(f"| {markdown_escape(item)} | {markdown_escape(requirement)} |\n" for item, requirement in rows).rstrip("\n")


def sample_binary_name(samples_root: Path, sample_dir: Path) -> str:
    return folder_name_for(samples_root, sample_dir)


def build_section(samples_root: Path, sample_dir: Path, language: str) -> str:
    rel = sample_dir.relative_to(samples_root)
    rel_posix = f"{samples_root.name}/{rel.as_posix()}"
    rel_ps = powershell_path(f"{samples_root.name}/{rel.as_posix()}")
    binary = sample_binary_name(samples_root, sample_dir)
    norm_lang = normalize_language(language)

    if norm_lang in {"c", "cpp"}:
        return (
            "Build from the sample folder using CMake directly, or use `scripts/build.py`/`build.ps1` "
            "at the repo root to build all samples in one shot.\n\n"
            "### CMake directly — Linux\n\n"
            "```bash\n"
            f"cd {rel_posix}\n"
            "cmake -B .cmake-tmp\n"
            "cmake --build .cmake-tmp\n"
            "```\n\n"
            "### CMake directly — Windows (PowerShell)\n\n"
            "```powershell\n"
            f"cd {rel_ps}\n"
            "cmake -B .cmake-tmp -A x64\n"
            "cmake --build .cmake-tmp --config Release\n"
            "```\n\n"
            "Binary lands in `.cmake-tmp/build/`."
        )

    if norm_lang == "csharp":
        csproj = next((path.name for path in sorted(sample_dir.glob("*.csproj"))), "<project>.csproj")
        return (
            "### Using scripts/build.py / build.ps1 (builds all samples)\n\n"
            "```powershell\n"
            "cd <repo-root>\n"
            "python scripts/build.py\n"
            "# or: .\\build.ps1\n"
            "```\n\n"
            "Binary and supporting files land in:\n\n"
            "```\n"
            f"build\\{binary}\\\n"
            "```\n\n"
            "### Directly with dotnet\n\n"
            "```powershell\n"
            f"cd {rel_ps}\n"
            f"dotnet build {csproj} -c Release --output .dotnet-tmp\n"
            "```\n\n"
            "Binary lands in `.dotnet-tmp\\` inside the sample folder."
        )

    if norm_lang == "python":
        return "No build step is required for this Python sample."

    return "TODO: describe how to build this sample."


def run_section(samples_root: Path, sample_dir: Path, language: str) -> str:
    rel = sample_dir.relative_to(samples_root)
    rel_posix = f"{samples_root.name}/{rel.as_posix()}"
    rel_ps = powershell_path(f"{samples_root.name}/{rel.as_posix()}")
    binary = sample_binary_name(samples_root, sample_dir)
    norm_lang = normalize_language(language)

    if norm_lang in {"c", "cpp"}:
        return (
            "After a direct CMake build:\n\n"
            "```bash\n"
            "# Linux\n"
            f".cmake-tmp/build/{binary}\n\n"
            "# Windows PowerShell\n"
            f".\\.cmake-tmp\\build\\{binary}.exe\n"
            "```"
        )

    if norm_lang == "csharp":
        exe_name = f"{Path(binary).name}.exe"
        csproj = next((path.name for path in sorted(sample_dir.glob("*.csproj"))), "<project>.csproj")
        return (
            "### After scripts/build.py / build.ps1\n\n"
            "```powershell\n"
            f".\\build\\{binary}\\{exe_name}\n"
            "```\n\n"
            "### After a direct dotnet build\n\n"
            "```powershell\n"
            f".\\{rel_ps}\\.dotnet-tmp\\{exe_name}\n"
            "```\n\n"
            "Or use `dotnet run` (no separate build step needed):\n\n"
            "```powershell\n"
            f"cd {rel_ps}\n"
            f"dotnet run --project {csproj}\n"
            "```"
        )

    if norm_lang == "python":
        return (
            "### From source\n\n"
            "```bash\n"
            f"cd {rel_posix}\n"
            "python main.py\n"
            "```\n\n"
            "### After scripts/build.py / build.ps1\n\n"
            "```powershell\n"
            f".\\build\\{binary}\\run.ps1\n"
            "```"
        )

    return "TODO: describe how to run this sample."


def notes_section(info: SampleInfo) -> str:
    notes = []
    if clean(info.note):
        notes.append(clean(info.note))
    if normalize_language(info.language) == "csharp":
        notes.append("Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.")
    if not notes:
        return "-"
    return "\n".join(f"- {note}" for note in notes)


def links_section(info: SampleInfo, language: str) -> str:
    links = list(LINKS_BY_LANGUAGE.get(language, []))
    if not links and normalize(info.api_type) == "xiapi":
        links.append("[xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)")
    if "[XIMEA Software Packages](https://www.ximea.com/software-downloads)" not in links:
        links.append("[XIMEA Software Packages](https://www.ximea.com/software-downloads)")
    return "\n".join(f"- {link}" for link in links)


class MissingKeyDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_readme(template: str, samples_root: Path, sample_dir: Path, info: SampleInfo, language: str) -> str:
    libraries = ", ".join(split_list(info.libraries)) or "-"
    values = MissingKeyDict(
        sample_title=title_for(info),
        sample_name=display(info.name),
        description=display(info.description, "TODO: add a short sample description."),
        category=markdown_escape(info.category),
        duplicate_group_id=markdown_escape(info.duplicate_group_id),
        os_platform=markdown_escape(info.os_platform),
        hardware_platform=markdown_escape(info.hardware_platform),
        api_type=markdown_escape(info.api_type),
        language=display(language),
        csv_language=markdown_escape(info.language),
        libraries=markdown_escape(libraries),
        author=markdown_escape(info.author),
        year=markdown_escape(info.year),
        location=markdown_escape(info.location),
        source_path=markdown_escape(info.path),
        copyright=markdown_escape(info.copyright),
        note=markdown_escape(info.note),
        done=markdown_escape(info.done),
        sdk_requirement=markdown_escape(sdk_requirement(info)),
        extra_prerequisites=extra_prerequisites(info, language),
        build_section=build_section(samples_root, sample_dir, language),
        run_section=run_section(samples_root, sample_dir, language),
        notes_section=notes_section(info),
        links_section=links_section(info, language),
    )
    rendered = template.format_map(values).rstrip() + "\n"
    return rendered


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Generate sample README.md files from a template and samples CSV.")
    parser.add_argument("--csv", type=Path, default=root / CSV_NAME, help="Path to samples CSV file.")
    parser.add_argument("--samples", type=Path, default=None, help="Path to Samples/ or samples/ directory.")
    parser.add_argument("--template", type=Path, default=root / DEFAULT_TEMPLATE, help="Markdown template path.")
    parser.add_argument("--write", action="store_true", help="Write README.md files. Default is dry-run.")
    parser.add_argument("--check", action="store_true", help="Fail if any README.md would change.")
    parser.add_argument("--sample-dir", action="append", type=Path, help="Only process this sample directory; can be repeated.")
    return parser.parse_args()


def resolve_sample_dir(path: Path, root: Path, samples_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == samples_root.name.lower():
        return (samples_root.joinpath(*parts[1:])).resolve()
    return (root / path).resolve()


def main() -> int:
    args = parse_args()
    root = repo_root()
    csv_path = args.csv.resolve()
    samples_root = args.samples.resolve() if args.samples else find_samples_root(root).resolve()
    template_path = args.template.resolve()

    samples = read_samples(csv_path)
    template = template_path.read_text(encoding="utf-8")

    sample_dirs = (
        [resolve_sample_dir(path, root, samples_root) for path in args.sample_dir]
        if args.sample_dir
        else iter_sample_dirs(samples_root)
    )
    changed = 0
    skipped = 0

    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            print(f"SKIP missing directory: {sample_dir}")
            skipped += 1
            continue
        language = language_for_dir(sample_dir)
        if language is None:
            print(f"SKIP no supported sample source/project file: {sample_dir}")
            skipped += 1
            continue
        try:
            info = find_sample_info(sample_dir, samples_root, samples, language)
        except ValueError as exc:
            print(f"SKIP outside samples root: {sample_dir} ({exc})")
            skipped += 1
            continue
        if info is None:
            print(f"SKIP no CSV row match: {sample_dir}")
            skipped += 1
            continue

        target = readme_path(sample_dir)
        old = target.read_text(encoding="utf-8-sig", errors="replace") if target.exists() else ""
        new = render_readme(template, samples_root, sample_dir, info, language)
        did_change = old != new
        changed += int(did_change)

        if args.write and did_change:
            target.write_text(new, encoding="utf-8", newline="\n")
            status = "UPDATED"
        elif args.write:
            status = "OK"
        else:
            status = "WOULD UPDATE" if did_change else "OK"

        print(f"{status}: {target} <- {info.name} ({info.language or language})")

    print(f"Summary: {len(sample_dirs)} directories processed, {changed} change(s), {skipped} skipped")
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
