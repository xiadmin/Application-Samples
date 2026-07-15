#!/usr/bin/env python3
"""Generate intro comments for C/C++/C#/Python sample files from the samples CSV."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from common import CSV_NAME, clean, normalize, normalize_language, read_sample_rows, repo_root

SUPPORTED_FILES = {"main.c": "C", "main.cpp": "C++", "Program.cs": "C#", "main.py": "Python"}
TEMPLATE_DIR = repo_root() / "scripts" / "templates"
TEMPLATE_BY_KIND = {
    "c": "intro-c-comment.txt",
    "slash": "intro-slash-comment.txt",
    "python": "intro-python-docstring.txt",
}


@dataclass(frozen=True)
class SampleInfo:
    name: str
    author: str
    category: str
    description: str
    os_platform: str
    hardware_platform: str
    api_type: str
    language: str
    libraries: str
    year: str
    copyright: str
    note: str


def read_samples(csv_path: Path) -> list[SampleInfo]:
    return [
        SampleInfo(
            name=data.get("Sample name", ""),
            author=data.get("Author", ""),
            category=data.get("Tomas_category", ""),
            description=data.get("Description", ""),
            os_platform=data.get("OS platform", ""),
            hardware_platform=data.get("Hardware platform", ""),
            api_type=data.get("API type", ""),
            language=data.get("Programming language", ""),
            libraries=data.get("Libraries", ""),
            year=data.get("Year of last update", ""),
            copyright=data.get("Copyright", ""),
            note=data.get("Note", ""),
        )
        for data in read_sample_rows(csv_path)
    ]


def path_tokens(source_file: Path, samples_root: Path) -> set[str]:
    rel = source_file.relative_to(samples_root)
    tokens = {normalize(part) for part in rel.parts[:-1]}
    tokens.add(normalize(rel.parent.name))
    tokens.add(normalize(source_file.stem))
    if rel.parent.name in {"c", "cpp"} and rel.parent.parent != samples_root:
        tokens.add(normalize(rel.parent.parent.name))
    return {token for token in tokens if token}


def row_score(row: SampleInfo, source_file: Path, samples_root: Path, language: str) -> int:
    tokens = path_tokens(source_file, samples_root)
    sample_name = normalize(row.name)
    if sample_name not in tokens:
        return -1

    score = 100
    if normalize_language(row.language) == normalize_language(language):
        score += 30

    rel_text = normalize(str(source_file.relative_to(samples_root)))
    api = normalize(row.api_type)
    if api and api in rel_text:
        score += 10
    if row.hardware_platform and normalize(row.hardware_platform) in rel_text:
        score += 5
    if row.os_platform and normalize(row.os_platform) in rel_text:
        score += 3
    return score


def find_sample_info(source_file: Path, samples_root: Path, samples: list[SampleInfo], language: str) -> SampleInfo | None:
    scored = [(row_score(row, source_file, samples_root, language), row) for row in samples]
    scored = [(score, row) for score, row in scored if score >= 0]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def wrap_text(value: str, *, width: int = 78, indent: str = "") -> list[str]:
    value = clean(value)
    if not value:
        return []

    available = width - len(indent)
    words = value.split()
    lines: list[str] = []
    current = ""

    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= available:
            current += " " + word
        else:
            lines.append(indent + current)
            current = word
    if current:
        lines.append(indent + current)
    return lines


def build_note(language: str) -> str:
    if normalize_language(language) == "python":
        return "Build: no build step needed — run directly with Python."
    return "Build: see README.md, scripts/build.py, or build.ps1 at the repo root."


def title_for(info: SampleInfo, language: str) -> str:
    normalized = normalize_language(language)
    if normalized == "cpp":
        return f"{info.name} - XIMEA xiAPIplus sample (C++17)"
    if normalized == "python":
        return f"{info.name} - XIMEA xiAPI capture sample (Python 3.9+)"
    return f"{info.name} - XIMEA xiAPI sample ({language})"


def render_from_template(kind: str, info: SampleInfo, language: str) -> str:
    template = (TEMPLATE_DIR / TEMPLATE_BY_KIND[kind]).read_text(encoding="utf-8")
    comment_prefix = " * " if kind == "c" else ""
    description = "\n".join(wrap_text(info.description or "TODO: describe what this sample does.", indent=comment_prefix))
    return template.format(
        title=title_for(info, language),
        description=description,
        build_note=build_note(language),
    ).rstrip() + "\n\n"


def render_intro(path: Path, info: SampleInfo, language: str) -> str:
    if path.suffix == ".c":
        return render_from_template("c", info, language)
    if path.suffix == ".py":
        return render_from_template("python", info, language)
    return render_from_template("slash", info, language)


def prefix_length(text: str) -> int:
    offset = 0
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        offset += len(lines[0])
        lines = lines[1:]
    if lines and re.match(r"#.*coding[:=]\s*[-\w.]+", lines[0]):
        offset += len(lines[0])
    return offset


def strip_existing_intro(text: str) -> str:
    start = prefix_length(text)
    prefix = text[:start]
    body = text[start:].lstrip("\ufeff\r\n")

    if body.startswith("/*"):
        end = body.find("*/")
        if end != -1:
            return prefix + body[end + 2 :].lstrip("\r\n")

    if body.startswith("//"):
        lines = body.splitlines(keepends=True)
        idx = 0
        while idx < len(lines) and (lines[idx].startswith("//") or not lines[idx].strip()):
            idx += 1
        return prefix + "".join(lines[idx:]).lstrip("\r\n")

    for quote in ('"""', "'''"):
        if body.startswith(quote):
            end = body.find(quote, len(quote))
            if end != -1:
                return prefix + body[end + len(quote) :].lstrip("\r\n")

    return text


def desired_content(path: Path, info: SampleInfo, language: str) -> str:
    old = path.read_text(encoding="utf-8-sig", errors="replace")
    stripped = strip_existing_intro(old)
    start = prefix_length(stripped)
    prefix = stripped[:start]
    body = stripped[start:].lstrip("\ufeff\r\n")
    return prefix + render_intro(path, info, language) + body


def update_file(path: Path, info: SampleInfo, language: str) -> bool:
    old = path.read_text(encoding="utf-8-sig", errors="replace")
    new = desired_content(path, info, language)
    if new == old:
        return False
    path.write_text(new, encoding="utf-8", newline="\n")
    return True


def language_for(path: Path) -> str | None:
    return SUPPORTED_FILES.get(path.name)


def iter_sample_files(samples_root: Path) -> list[Path]:
    return sorted(path for path in samples_root.rglob("*") if path.is_file() and language_for(path) is not None)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Generate intro comments for C/C++/C#/Python sample files from the samples CSV.")
    parser.add_argument("--csv", type=Path, default=root / CSV_NAME, help="Path to samples CSV file.")
    parser.add_argument("--samples", type=Path, default=root / "samples", help="Path to samples directory.")
    parser.add_argument("--write", action="store_true", help="Write changes. Default is dry-run.")
    parser.add_argument("--check", action="store_true", help="Fail if any file would change.")
    parser.add_argument("--file", action="append", type=Path, help="Only process this sample source file; can be repeated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.csv = args.csv.resolve()
    args.samples = args.samples.resolve()
    samples = read_samples(args.csv)
    files = [path.resolve() for path in args.file] if args.file else iter_sample_files(args.samples)

    changed = 0
    skipped = 0
    for path in files:
        language = language_for(path)
        if language is None:
            print(f"SKIP unsupported file: {path}")
            skipped += 1
            continue

        info = find_sample_info(path, args.samples, samples, language)
        if info is None:
            print(f"SKIP no CSV row match: {path}")
            skipped += 1
            continue

        if args.write:
            did_change = update_file(path, info, language)
            status = "UPDATED" if did_change else "OK"
        else:
            old = path.read_text(encoding="utf-8-sig", errors="replace")
            did_change = desired_content(path, info, language) != old
            status = "WOULD UPDATE" if did_change else "OK"

        changed += int(did_change)
        print(f"{status}: {path} <- {info.name} ({info.language})")

    print(f"Summary: {len(files)} file(s), {changed} change(s), {skipped} skipped")
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
