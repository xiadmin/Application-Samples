#!/usr/bin/env python3
"""Run built samples and require the expected no-camera failure."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NO_CAMERA_MESSAGE = "Error: no XIMEA cameras detected"


@dataclass(frozen=True)
class SampleCommand:
    name: str
    argv: list[str]
    cwd: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return parsed


def is_executable(path: Path) -> bool:
    if path.suffix.lower() == ".exe":
        return True
    return os.name != "nt" and path.suffix == "" and os.access(path, os.X_OK)


def discover_commands(build_root: Path, python_roots: list[Path]) -> list[SampleCommand]:
    commands = [
        SampleCommand(path.parent.name, [str(path.resolve())], path.parent.resolve())
        for path in sorted(build_root.rglob("*"))
        if path.is_file() and is_executable(path)
    ]
    for python_root in python_roots:
        commands.extend(
            SampleCommand(path.parent.name, [sys.executable, str(path.resolve())], path.parent.resolve())
            for path in sorted(python_root.rglob("main.py"))
        )
    return commands


def run_sample(command: SampleCommand, timeout_seconds: int) -> str | None:
    print(f"Running {command.name}: {subprocess.list2cmdline(command.argv)}", flush=True)
    try:
        completed = subprocess.run(
            command.argv,
            cwd=command.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)

    output = completed.stdout + completed.stderr
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    if completed.returncode == 0:
        return "exited successfully instead of reporting that no camera was detected"
    if NO_CAMERA_MESSAGE not in output:
        return f"exited with code {completed.returncode} without the expected no-camera error"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all built samples and verify that each fails only because no camera is connected."
    )
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, action="append", default=[])
    parser.add_argument("--timeout-seconds", type=positive_int, default=90)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_root = args.build_root.resolve()
    python_roots = [path.resolve() for path in args.python_root]

    if not build_root.is_dir():
        print(f"Build root does not exist: {build_root}", file=sys.stderr)
        return 1

    missing_python_roots = [path for path in python_roots if not path.is_dir()]
    if missing_python_roots:
        for path in missing_python_roots:
            print(f"Python sample root does not exist: {path}", file=sys.stderr)
        return 1

    commands = discover_commands(build_root, python_roots)
    if not commands:
        print(f"No runnable samples found under {build_root}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for command in commands:
        failure = run_sample(command, args.timeout_seconds)
        if failure:
            failures.append(f"{command.name}: {failure}")

    if failures:
        print("Unexpected sample runtime results:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"All {len(commands)} sample(s) reported the expected no-camera error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
