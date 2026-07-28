#!/usr/bin/env python3
"""Generate intro comments for C/C++/C#/Python sample files."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from common import clean, find_samples_root, normalize, normalize_language, read_sample_rows, repo_root

SUPPORTED_FILES = {"main.c": "C", "main.cpp": "C++", "Program.cs": "C#", "main.py": "Python"}
TODO_TEXT = "TODO"
TEMPLATE_DIR = repo_root() / "scripts" / "templates"
TEMPLATE_BY_KIND = {
    "block": "intro-block-comment.txt",
    "python": "intro-python-docstring.txt",
}
COPYRIGHT_TEXT = """Copyright (c) 2026 XIMEA s.r.o.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE."""


@dataclass(frozen=True)
class SampleInfo:
    name: str
    category: str
    description: str
    os_platform: str
    hardware_platform: str
    api_type: str
    language: str
    libraries: str
    copyright: str


def read_samples(csv_path: Path) -> list[SampleInfo]:
    return [
        SampleInfo(
            name=data.get("Sample name", ""),
            category=data.get("Tomas_category", ""),
            description=data.get("Description", ""),
            os_platform=data.get("OS platform", ""),
            hardware_platform=data.get("Hardware platform", ""),
            api_type=data.get("API type", ""),
            language=data.get("Programming language", ""),
            libraries=data.get("Libraries", ""),
            copyright=data.get("Copyright", ""),
        )
        for data in read_sample_rows(csv_path)
    ]


def default_sample_info(source_file: Path, samples_root: Path, language: str) -> SampleInfo:
    return SampleInfo(
        name=source_file.parent.name,
        category="",
        description="",
        os_platform=os_platform_for_path(source_file, samples_root),
        hardware_platform="",
        api_type=api_type_for_path(source_file, samples_root, ""),
        language=language,
        libraries="",
        copyright="",
    )


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
    return "Build: see README.md or scripts/build.py at the repo root."


def title_for(info: SampleInfo, language: str) -> str:
    normalized = normalize_language(language)
    if normalized == "cpp":
        return f"{info.name} - XIMEA xiAPIplus sample (C++17)"
    if normalized == "python":
        return f"{info.name} - XIMEA xiAPI capture sample (Python 3.11+)"
    return f"{info.name} - XIMEA xiAPI sample ({language})"


def api_type_for_path(path: Path, samples_root: Path, fallback: str) -> str:
    rel_parts = path.relative_to(samples_root).parts
    api_folder = rel_parts[0] if rel_parts else ""
    api_types = {
        "xiapi": "xiAPI",
        "xiapiplus": "xiAPIplus",
        "xiapi-net-csharp": "xiAPI.NET",
        "xiapi-python": "xiAPI Python",
    }
    return api_types.get(api_folder, fallback)


def os_platform_for_path(path: Path, samples_root: Path) -> str:
    rel_parts = path.relative_to(samples_root).parts
    normalized_parts = {normalize(part) for part in rel_parts[:-1]}
    api_folder = normalize(rel_parts[0]) if rel_parts else ""

    if api_folder == "xiapinetcsharp":
        return "Windows"
    if "crossplatform" in normalized_parts:
        return "Cross-platform"
    if any("windows" in part for part in normalized_parts):
        return "Windows"
    if any(part in {"macos", "osx", "darwin"} for part in normalized_parts):
        return "macOS"
    if any("linux" in part or "jetson" in part for part in normalized_parts):
        return "Linux"
    return ""


def field_value(value: str) -> str:
    return clean(value) or TODO_TEXT


def format_field(label: str, value: str, *, prefix: str) -> str:
    return "\n".join(wrap_text(f"{label}: {field_value(value)}", indent=prefix))


def format_block(value: str, *, prefix: str) -> str:
    lines: list[str] = []
    for raw_line in value.splitlines():
        if raw_line:
            lines.append(prefix + raw_line)
        else:
            lines.append(prefix.rstrip())
    return "\n".join(lines)


def render_from_template(kind: str, info: SampleInfo, language: str, api_type: str) -> str:
    template = (TEMPLATE_DIR / TEMPLATE_BY_KIND[kind]).read_text(encoding="utf-8")
    comment_prefix = " * " if kind == "block" else ""
    return template.format(
        sample_name=format_field("Sample name", info.name, prefix=comment_prefix),
        category=format_field("Category", info.category, prefix=comment_prefix),
        os_platform=format_field("OS platform", info.os_platform, prefix=comment_prefix),
        hardware_platform=format_field("Hardware platform", info.hardware_platform, prefix=comment_prefix),
        api_type=format_field("API type", api_type, prefix=comment_prefix),
        short_description=format_field("Short description", info.description, prefix=comment_prefix),
        copyright=format_block(COPYRIGHT_TEXT, prefix=comment_prefix),
    ).rstrip() + "\n\n"


def render_intro(path: Path, samples_root: Path, info: SampleInfo, language: str) -> str:
    api_type = api_type_for_path(path.resolve(), samples_root, info.api_type)
    if path.suffix == ".py":
        return render_from_template("python", info, language, api_type)
    return render_from_template("block", info, language, api_type)


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
        while idx < len(lines) and lines[idx].startswith("//"):
            idx += 1
        return prefix + "".join(lines[idx:]).lstrip("\r\n")

    for quote in ('"""', "'''"):
        if body.startswith(quote):
            end = body.find(quote, len(quote))
            if end != -1:
                return prefix + body[end + len(quote) :].lstrip("\r\n")

    return text


def has_existing_intro(text: str) -> bool:
    start = prefix_length(text)
    body = text[start:].lstrip("\ufeff\r\n")

    if body.startswith("/*"):
        return body.find("*/") != -1

    if body.startswith("//"):
        return True

    return any(body.startswith(quote) and body.find(quote, len(quote)) != -1 for quote in ('"""', "'''"))


def desired_content(path: Path, samples_root: Path, info: SampleInfo, language: str) -> str:
    old = path.read_text(encoding="utf-8-sig", errors="replace")
    if has_existing_intro(old):
        old = strip_existing_intro(old)

    start = prefix_length(old)
    prefix = old[:start]
    body = old[start:].lstrip("\ufeff\r\n")
    return prefix + render_intro(path, samples_root, info, language) + body


def update_file(path: Path, samples_root: Path, info: SampleInfo, language: str) -> bool:
    old = path.read_text(encoding="utf-8-sig", errors="replace")
    new = desired_content(path, samples_root, info, language)
    if new == old:
        return False
    path.write_text(new, encoding="utf-8", newline="\n")
    return True


def language_for(path: Path) -> str | None:
    return SUPPORTED_FILES.get(path.name)


def iter_sample_files(samples_root: Path) -> list[Path]:
    return sorted(path for path in samples_root.rglob("*") if path.is_file() and language_for(path) is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate intro comments for C/C++/C#/Python sample files.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Populate the generated intro metadata from this CSV.",
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        help="Only process this sample source file. Defaults to all supported sample entry files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    samples_root = find_samples_root(root).resolve()
    csv_path = args.csv_path.resolve() if args.csv_path else None
    samples = read_samples(csv_path) if csv_path else []
    files = [args.file_path.resolve()] if args.file_path else iter_sample_files(samples_root)

    changed = 0
    skipped = 0
    for path in files:
        if not path.is_file():
            print(f"SKIP missing file: {path}")
            skipped += 1
            continue

        language = language_for(path)
        if language is None:
            print(f"SKIP unsupported file: {path}")
            skipped += 1
            continue

        if not path.is_relative_to(samples_root):
            print(f"SKIP file outside samples root: {path}")
            skipped += 1
            continue

        info = (
            find_sample_info(path, samples_root, samples, language)
            if csv_path
            else default_sample_info(path, samples_root, language)
        )
        if info is None:
            print(f"SKIP no CSV row match: {path}")
            skipped += 1
            continue

        did_change = update_file(path, samples_root, info, language)
        status = "UPDATED" if did_change else "OK"

        changed += int(did_change)
        print(f"{status}: {path} <- {info.name} ({info.language})")

    print(f"Summary: {len(files)} file(s), {changed} change(s), {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
