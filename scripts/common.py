#!/usr/bin/env python3
"""Shared helpers for maintenance scripts."""

from __future__ import annotations

import csv
import re
from pathlib import Path

CSV_NAME = "ximea-samples.csv"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_samples_root(root: Path, *, create: bool = False) -> Path:
    for name in ("samples", "Samples"):
        candidate = root / name
        if candidate.is_dir():
            return candidate

    if create:
        candidate = root / "samples"
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    raise FileNotFoundError(f"Could not find samples/ under {root}")


def clean(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_language(value: str) -> str:
    value = clean(value).lower().replace(" ", "")
    if value in {"c++", "cpp", "cxx"}:
        return "cpp"
    if value in {"c#", "csharp", "cs"}:
        return "csharp"
    if value in {"python", "py", "python3"}:
        return "python"
    if value == "c":
        return "c"
    return normalize(value)


def split_list(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", clean(value)) if part.strip()]


def read_sample_rows(csv_path: Path) -> list[dict[str, str]]:
    text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    rows = csv.reader(text.splitlines(), delimiter=";")
    header: list[str] | None = None
    records: list[dict[str, str]] = []

    for row in rows:
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0].strip() == "Sample name":
            header = [cell.strip() for cell in row]
            continue
        if header is None:
            continue

        data = {header[i]: clean(row[i]) if i < len(row) else "" for i in range(len(header))}
        if data.get("Sample name"):
            records.append(data)

    if not records:
        raise ValueError(f"No sample rows found in {csv_path}")
    return records


def folder_name_for(samples_root: Path, sample_dir: Path) -> str:
    return "-".join(sample_dir.relative_to(samples_root).parts)


def powershell_path(path: str) -> str:
    return path.replace("/", "\\")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
