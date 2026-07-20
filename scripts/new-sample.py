#!/usr/bin/env python3
"""Create a new Application-Samples scaffold."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import CSV_NAME, clean, find_samples_root, repo_root, write_text as write

INVALID_NAME = re.compile(r'[\\/:*?"<>|]')
INVALID_PATH_CHARS = re.compile(r'[:*?"<>|]')
KEBAB_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TEMPLATE_DIR = repo_root() / "scripts" / "templates"
LANGUAGES = ["c", "cpp", "csharp", "python"]
COPYRIGHT_TEXT = """Copyright (c) 2026 XIMEA s.r.o.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE."""


def valid_folder_name(name: str) -> bool:
    return bool(name) and name == name.strip() and INVALID_NAME.search(name) is None and KEBAB_NAME.fullmatch(name) is not None


def ask_name(prompt: str) -> str:
    while True:
        value = input(f"{prompt}: ").strip()
        if valid_folder_name(value):
            return value
        print('Invalid name. Use lowercase kebab-case and do not use \\ / : * ? " < > |.')


def choose(items: list[str], prompt: str, allow_new: bool = True) -> str | None:
    all_items = list(items)
    if allow_new:
        all_items.append("+ Create new")

    while True:
        print()
        print(prompt)
        for i, item in enumerate(all_items, start=1):
            print(f"  {i}. {item}")
        raw = input("Select number: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= idx <= len(all_items):
            if allow_new and idx == len(all_items):
                return None
            return all_items[idx - 1]
        print("Selection out of range.")


def print_tree(root: Path, *, max_depth: int = 4) -> None:
    print(f"{root.name}/")

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        children = sorted(p for p in directory.iterdir() if p.is_dir())
        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{child.name}/")
            next_prefix = prefix + ("    " if is_last else "│   ")
            walk(child, next_prefix, depth + 1)

    walk(root, "", 0)


def normalize_relative_sample_path(value: str) -> str:
    return value.strip().replace("\\", "/").strip("/")


def validate_sample_path(samples_dir: Path, value: str) -> str | None:
    if not value:
        return "Path is required."
    if INVALID_PATH_CHARS.search(value) is not None:
        return 'Invalid path. Do not use : * ? " < > |.'

    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        return "Invalid path. Use folder names separated by / or \\, without empty, . or .. segments."

    current = samples_dir
    for part in parts:
        current = current / part
        if current.exists():
            if not current.is_dir():
                return f"Invalid path. Existing segment is not a folder: {part}"
            continue
        if not valid_folder_name(part):
            return f"Invalid new folder name: {part}. Use lowercase kebab-case."
    return None


def ask_sample_path(samples_dir: Path) -> str:
    while True:
        print()
        print("Current samples folder structure:")
        print_tree(samples_dir)
        print()
        print("Enter the new sample folder path relative to samples/.")
        print("Examples: xiapi/cross-platform/capture-50-images, opencv\\capture-and-process")
        value = normalize_relative_sample_path(input("New sample path: "))
        error = validate_sample_path(samples_dir, value)
        if error is None:
            return value
        print(error)


def infer_language_from_path(parts: list[str]) -> str | None:
    if not parts:
        return None
    api = parts[0]
    if api == "xiapi":
        return "c"
    if api == "xiapiplus":
        return "cpp"
    if api == "xiapi.net-c#":
        return "csharp"
    if api == "xiapi-python":
        return "python"
    return None


def validate_language_for_path(parts: list[str], lang: str) -> str | None:
    inferred = infer_language_from_path(parts)
    if inferred is not None and lang != inferred:
        api = parts[0]
        return f"Error: {api} samples must use lang={inferred}."
    return None


def pascal_case(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^a-zA-Z0-9]+", value) if part)


def target_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", value)


def render_template(name: str, **values: str) -> str:
    text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
        text = text.replace("{" + key + "}", value)
    return text


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


def format_intro_field(label: str, value: str, *, prefix: str) -> str:
    return "\n".join(wrap_text(f"{label}: {clean(value) or 'TODO'}", indent=prefix))


def format_intro_block(value: str, *, prefix: str) -> str:
    lines: list[str] = []
    for raw_line in value.splitlines():
        if raw_line:
            lines.append(prefix + raw_line)
        else:
            lines.append(prefix.rstrip())
    return "\n".join(lines)


def render_intro_comment(lang: str, sample_name: str, api_type: str) -> str:
    if lang == "c":
        template = "intro-c-comment.txt"
        prefix = " * "
    elif lang == "python":
        template = "intro-python-docstring.txt"
        prefix = ""
    else:
        template = "intro-slash-comment.txt"
        prefix = "// "

    return render_template(
        template,
        sample_name=format_intro_field("Sample name", sample_name, prefix=prefix),
        category=format_intro_field("Category", "TODO", prefix=prefix),
        os_platform=format_intro_field("OS platform", "TODO", prefix=prefix),
        hardware_platform=format_intro_field("Hardware platform", "TODO", prefix=prefix),
        api_type=format_intro_field("API type", api_type, prefix=prefix),
        short_description=format_intro_field("Short description", "TODO", prefix=prefix),
        copyright=format_intro_block(COPYRIGHT_TEXT, prefix=prefix),
    ).rstrip()


def display_api_type(api_folder: str) -> str:
    api_types = {
        "xiapi": "xiAPI",
        "xiapiplus": "xiAPIplus",
        "xiapi.net-c#": "xiAPI.NET",
        "xiapi-python": "xiAPI Python",
    }
    return api_types.get(api_folder, api_folder)


def make_csharp(sample_dir: Path, binary_name: str, cs_project_name: str, intro_comment: str) -> Path:
    write(
        sample_dir / f"{cs_project_name}.csproj",
        render_template("scaffold-csharp-csproj.xml", binary_name=binary_name),
    )
    source_file = sample_dir / "Program.cs"
    write(source_file, render_template("scaffold-csharp-program.cs", binary_name=binary_name, intro_comment=intro_comment))
    return source_file


def make_python(sample_dir: Path, binary_name: str, intro_comment: str) -> Path:
    source_file = sample_dir / "main.py"
    write(source_file, render_template("scaffold-python-main.py", binary_name=binary_name, intro_comment=intro_comment))
    return source_file


def make_cmake(sample_dir: Path, lang: str, binary_name: str, cmake_include_path: str, intro_comment: str) -> Path:
    source_file = "main.c" if lang == "c" else "main.cpp"
    template_file = "scaffold-c-main.c" if lang == "c" else "scaffold-cpp-main.cpp"
    lang_std = "c_std_11" if lang == "c" else "cxx_std_17"
    project_lang = "C" if lang == "c" else "CXX"
    ximea_target = "XIMEA::xiAPI" if lang == "c" else "XIMEA::xiAPIplus"
    target = target_name(binary_name)

    write(
        sample_dir / "CMakeLists.txt",
        render_template(
            "scaffold-cmake-cmakelists.txt",
            binary_name=binary_name,
            cmake_include_path=cmake_include_path,
            lang_std=lang_std,
            project_lang=project_lang,
            source_file=source_file,
            target=target,
            ximea_target=ximea_target,
        ),
    )
    source_path = sample_dir / source_file
    write(source_path, render_template(template_file, binary_name=binary_name, intro_comment=intro_comment))
    return source_path


def run_generator(root: Path, args: list[str]) -> bool:
    sys.stdout.flush()
    result = subprocess.run([sys.executable, *args], cwd=root)
    return result.returncode == 0


def run_metadata_generators(root: Path, samples_dir: Path, sample_dir: Path, *, use_csv: bool) -> bool:
    ok = True
    print()
    print("Generating README...")
    command = [
        str(root / "scripts" / "generate-readmes.py"),
        "--samples", str(samples_dir),
        "--write",
        "--sample-dir", str(sample_dir),
    ]
    if use_csv:
        csv_path = root / CSV_NAME
        if not csv_path.is_file():
            print(f"WARNING: {CSV_NAME} not found; generated README without CSV metadata.", file=sys.stderr)
        else:
            command.extend(["--use-csv", "--csv", str(csv_path)])
    ok &= run_generator(
        root,
        command,
    )
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new Application-Samples scaffold.")
    parser.add_argument("--path", help="Sample path relative to samples/, e.g. xiapi/cross-platform/capture-50-images")
    parser.add_argument("--api", help="API folder, e.g. xiapi, xiapiplus, xiapi.net-c#, xiapi-python")
    parser.add_argument("--group", choices=["cross-platform", "hardware-specific"], help="sample group for APIs that use grouped samples")
    parser.add_argument("--sample", help="Sample name/folder")
    parser.add_argument("--lang", choices=["c", "cpp", "csharp", "python"], help="Language/template")
    parser.add_argument("--use-csv", action="store_true", help="Populate generated README metadata from the samples CSV when available.")
    parser.add_argument("--yes", action="store_true", help="Create without confirmation when all required values are supplied.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    samples_dir = find_samples_root(root, create=True)

    if args.path:
        rel_path = normalize_relative_sample_path(args.path)
    elif args.api or args.group or args.sample:
        api = args.api or ask_name("API folder name")
        if api in {"xiapi", "xiapiplus", "xiapi-python"}:
            group = args.group or choose(["cross-platform", "hardware-specific"], "Select sample group:", allow_new=False)
            if group is None:
                print("No sample group selected.", file=sys.stderr)
                return 1
            sample = args.sample or ask_name("New sample name")
            rel_path = f"{api}/{group}/{sample}"
        else:
            sample = args.sample or ask_name("New sample name")
            rel_path = f"{api}/{sample}"
    else:
        rel_path = ask_sample_path(samples_dir)

    path_error = validate_sample_path(samples_dir, rel_path)
    if path_error is not None:
        print(path_error, file=sys.stderr)
        return 1

    parts = rel_path.split("/")
    topic = parts[-1]

    if args.lang:
        lang = args.lang
    elif args.yes:
        lang = infer_language_from_path(parts)
        if lang is None:
            print("Error: --lang is required with --yes for paths without a known API language.", file=sys.stderr)
            return 1
    else:
        lang = choose(LANGUAGES, "Select programming language/template:", allow_new=False)

    assert lang in set(LANGUAGES)

    lang_error = validate_language_for_path(parts, lang)
    if lang_error is not None:
        print(lang_error, file=sys.stderr)
        return 1

    sample_dir = samples_dir.joinpath(*parts)
    folder_name = "-".join(parts)
    sample_path = f"{samples_dir.name}/{rel_path}"

    cmake_depth = len(sample_path.split("/"))
    cmake_include_path = "/".join([".."] * cmake_depth) + "/cmake"
    binary_name = folder_name if lang in {"c", "cpp"} else f"{topic}-{lang}"
    cs_project_name = pascal_case(binary_name)
    api_type = display_api_type(parts[0])
    intro_comment = render_intro_comment(lang, topic, api_type)

    print()
    print("-----------------------------------------")
    print(f" Sample   : {topic}")
    print(f" Language : {lang}")
    print(f" Path     : {sample_path}")
    print(f" Binary   : {binary_name}")
    print("-----------------------------------------")

    if not args.yes:
        confirm = choose(["Yes", "No"], "Create this sample?", allow_new=False)
        if confirm != "Yes":
            print("Aborted.")
            return 0

    if sample_dir.exists():
        print(f"Error: {sample_path} already exists. Nothing was created.", file=sys.stderr)
        return 1

    sample_dir.mkdir(parents=True)
    if lang == "csharp":
        make_csharp(sample_dir, binary_name, cs_project_name, intro_comment)
    elif lang == "python":
        make_python(sample_dir, binary_name, intro_comment)
    else:
        make_cmake(sample_dir, lang, binary_name, cmake_include_path, intro_comment)

    if not run_metadata_generators(root, samples_dir, sample_dir, use_csv=args.use_csv):
        return 1

    print()
    print("Sample scaffold created:")
    for item in sorted(sample_dir.iterdir()):
        print(f"  {item.name}")
    print()
    print("Next steps:")
    if lang == "csharp":
        print("  1. Fill in the TODO sections in Program.cs")
        print("  2. Review generated README.md and replace TODO metadata")
        print(f"  3. Build: cd {sample_path} && dotnet build {cs_project_name}.csproj")
    elif lang == "python":
        print("  1. Fill in the TODO sections in main.py")
        print("  2. Review generated README.md and replace TODO metadata")
        print(f"  3. Run: python {sample_path}/main.py")
    else:
        source_name = "main.c" if lang == "c" else "main.cpp"
        print(f"  1. Fill in the TODO sections in {source_name}")
        print("  2. Review generated README.md and replace TODO metadata")
        print(f"  3. Build: cd {sample_path} && cmake -B .cmake-tmp && cmake --build .cmake-tmp")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
