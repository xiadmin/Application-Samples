#!/usr/bin/env python3
"""Build or check selected Application-Samples entries."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from common import find_samples_root, folder_name_for, repo_root


class SampleType(str, Enum):
    CMAKE = "cmake"
    DOTNET = "dotnet"
    PYTHON = "python"


class SampleStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    MISSING_DEPENDENCY = "missing_dependency"


@dataclass(frozen=True)
class Sample:
    name: str
    sample_type: SampleType
    path: Path
    entry: Path
    is_platform_specific: bool


class RuntimeStatus(str, Enum):
    PASSED = "passed"
    NO_CAMERA_OR_FAILED = "no_camera_or_failed"
    TIMED_OUT = "timed_out"
    COULD_NOT_START = "could_not_start"


@dataclass(frozen=True)
class RuntimeCommand:
    sample: Sample
    label: str
    argv: list[str]
    cwd: Path


@dataclass
class BuildResult:
    sample: Sample
    status: SampleStatus
    message: str = ""
    copied_files: int = 0
    runtime_commands: list[RuntimeCommand] | None = None


@dataclass
class RuntimeResult:
    command: RuntimeCommand
    status: RuntimeStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str = ""


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def remove_sample_work_dirs(sample_dir: Path) -> None:
    for name in (".cmake-tmp", ".dotnet-tmp", "__pycache__", "build", "bin", "obj"):
        remove_tree(sample_dir / name)


def run(cmd: list[str]) -> bool:
    print("+ " + " ".join(str(item) for item in cmd), flush=True)
    try:
        return subprocess.run(cmd).returncode == 0
    except FileNotFoundError as error:
        print(f"ERROR: command not found: {error.filename}", file=sys.stderr)
        return False


def find_runnable_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    runnable: list[Path] = []
    for item in sorted(directory.iterdir()):
        if not item.is_file():
            continue
        if item.suffix.lower() == ".exe" or os.access(item, os.X_OK):
            runnable.append(item)
    return runnable


def find_dotnet_command(sample: Sample, target_build_dir: Path) -> RuntimeCommand | None:
    exe_files = sorted(target_build_dir.glob("*.exe"))
    if exe_files:
        return RuntimeCommand(sample, exe_files[0].name, [str(exe_files[0])], target_build_dir)

    dll_files = sorted(
        path for path in target_build_dir.glob("*.dll")
        if not path.name.endswith(".resources.dll") and path.name != "xiApi.NETX64.dll"
    )
    if dll_files:
        return RuntimeCommand(sample, dll_files[0].name, ["dotnet", str(dll_files[0])], target_build_dir)
    return None


def runtime_commands_for_build_result(result: BuildResult) -> list[RuntimeCommand]:
    return result.runtime_commands or []


def timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def execute_runtime_command(command: RuntimeCommand, timeout_seconds: int) -> RuntimeResult:
    print("+ " + " ".join(str(item) for item in command.argv), flush=True)
    try:
        completed = subprocess.run(
            command.argv,
            cwd=command.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        return RuntimeResult(
            command,
            RuntimeStatus.COULD_NOT_START,
            message=f"command not found: {error.filename}",
        )
    except OSError as error:
        return RuntimeResult(command, RuntimeStatus.COULD_NOT_START, message=str(error))
    except subprocess.TimeoutExpired as error:
        return RuntimeResult(
            command,
            RuntimeStatus.TIMED_OUT,
            stdout=timeout_output_to_text(error.stdout),
            stderr=timeout_output_to_text(error.stderr),
            message=f"timed out after {timeout_seconds} seconds",
        )

    status = RuntimeStatus.PASSED if completed.returncode == 0 else RuntimeStatus.NO_CAMERA_OR_FAILED
    return RuntimeResult(
        command,
        status,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_runtime_commands(commands: list[RuntimeCommand], timeout_seconds: int) -> list[RuntimeResult]:
    return [execute_runtime_command(command, timeout_seconds) for command in commands]


def copy_files(src: Path, dst: Path) -> int:
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            count += 1
    return count


def copy_cmake_outputs(tmp_build_dir: Path, target_build_dir: Path, configuration: str) -> int:
    copied = 0
    built_bin_dir = tmp_build_dir / "build"
    built_config_dir = built_bin_dir / configuration

    if built_config_dir.is_dir():
        copied += copy_files(built_config_dir, target_build_dir)
    if built_bin_dir.is_dir():
        copied += copy_files(built_bin_dir, target_build_dir)

    if copied:
        return copied

    target_build_dir.mkdir(parents=True, exist_ok=True)
    for item in tmp_build_dir.rglob("*"):
        if not item.is_file():
            continue
        item_text = str(item)
        if "CMakeFiles" in item_text or item.name in {"Makefile", "cmake_install.cmake"}:
            continue
        if item.suffix == ".exe" or item.suffix == "":
            shutil.copy2(item, target_build_dir / item.name)
            copied += 1
    return copied


def is_platform_specific(sample_dir: Path) -> bool:
    return "cross-platform" not in sample_dir.parts


def discover_samples(samples_root: Path) -> list[Sample]:
    samples: dict[tuple[SampleType, Path], Sample] = {}

    for cmake_file in sorted(samples_root.rglob("CMakeLists.txt")):
        sample_dir = cmake_file.parent
        sample = Sample(
            name=folder_name_for(samples_root, sample_dir),
            sample_type=SampleType.CMAKE,
            path=sample_dir,
            entry=cmake_file,
            is_platform_specific=is_platform_specific(sample_dir),
        )
        samples[(sample.sample_type, sample.path)] = sample

    for csproj_file in sorted(samples_root.rglob("*.csproj")):
        sample_dir = csproj_file.parent
        sample = Sample(
            name=folder_name_for(samples_root, sample_dir),
            sample_type=SampleType.DOTNET,
            path=sample_dir,
            entry=csproj_file,
            is_platform_specific=is_platform_specific(sample_dir),
        )
        samples[(sample.sample_type, sample.path)] = sample

    for py_main in sorted(samples_root.rglob("main.py")):
        sample_dir = py_main.parent
        sample = Sample(
            name=folder_name_for(samples_root, sample_dir),
            sample_type=SampleType.PYTHON,
            path=sample_dir,
            entry=py_main,
            is_platform_specific=is_platform_specific(sample_dir),
        )
        samples[(sample.sample_type, sample.path)] = sample

    return sorted(samples.values(), key=lambda sample: (sample.sample_type.value, sample.name))


def normalize_selector(value: str, root: Path) -> str:
    raw = value.strip()
    try:
        path = Path(raw)
        if path.exists():
            return str(path.resolve())
        candidate = root / raw
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        pass
    return raw


def sample_matches_selector(sample: Sample, root: Path, selector: str) -> bool:
    sample_relative = sample.path.relative_to(root)
    return selector in {
        sample.name,
        str(sample.path),
        str(sample.path.resolve()),
        str(sample_relative),
        str(Path(root.name) / sample_relative),
    }


def dedupe_samples(samples: list[Sample]) -> list[Sample]:
    seen: set[tuple[SampleType, Path]] = set()
    deduped: list[Sample] = []
    for sample in samples:
        key = (sample.sample_type, sample.path)
        if key not in seen:
            seen.add(key)
            deduped.append(sample)
    return deduped


def resolve_sample_selectors(samples: list[Sample], root: Path, selectors: list[str]) -> tuple[list[Sample], list[str]]:
    matched: list[Sample] = []
    missing: list[str] = []

    for selector in selectors:
        normalized = normalize_selector(selector, root)
        selector_matches = [
            sample for sample in samples
            if sample_matches_selector(sample, root, normalized)
        ]
        if selector_matches:
            matched.extend(selector_matches)
        else:
            missing.append(selector)

    return dedupe_samples(matched), missing


def select_samples(samples: list[Sample], root: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str]]:
    selected = samples

    if args.type:
        allowed = {SampleType(value) for value in args.type}
        selected = [sample for sample in selected if sample.sample_type in allowed]

    missing: list[str] = []
    if args.sample:
        selected, missing = resolve_sample_selectors(selected, root, args.sample)

    if getattr(args, "skip_cmake", False):
        selected = [sample for sample in selected if sample.sample_type is not SampleType.CMAKE]
    if getattr(args, "skip_dotnet", False):
        selected = [sample for sample in selected if sample.sample_type is not SampleType.DOTNET]
    if getattr(args, "skip_python", False):
        selected = [sample for sample in selected if sample.sample_type is not SampleType.PYTHON]

    return selected, missing


def parse_selection(text: str, max_index: int) -> set[int]:
    selected: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))

    invalid = [index for index in selected if index < 1 or index > max_index]
    if invalid:
        raise ValueError(f"Selection out of range: {invalid}")
    return selected


def choose_samples_tui(samples: list[Sample], root: Path) -> list[Sample]:
    if not samples:
        return []

    print("Available samples:")
    for index, sample in enumerate(samples, start=1):
        marker = " platform-specific" if sample.is_platform_specific else ""
        rel = sample.path.relative_to(root)
        print(f"  {index:2d}. [{sample.sample_type.value}]{marker} {sample.name} ({rel})")

    while True:
        choice = input("Select samples: a=all, q=quit, numbers/ranges like 1,3-5: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return []
        if choice in {"a", "all", ""}:
            return samples
        try:
            selected_indexes = parse_selection(choice, len(samples))
        except (ValueError, TypeError) as error:
            print(f"Invalid selection: {error}")
            continue
        return [sample for index, sample in enumerate(samples, start=1) if index in selected_indexes]


def build_cmake_sample(sample: Sample, cmake_tmp_root: Path, final_build_root: Path, configuration: str) -> BuildResult:
    tmp_build_dir = cmake_tmp_root / sample.name
    target_build_dir = final_build_root / sample.name

    if shutil.which("cmake") is None:
        return BuildResult(sample, SampleStatus.MISSING_DEPENDENCY, "cmake CLI not found")

    if not run(["cmake", "-S", str(sample.path), "-B", str(tmp_build_dir)]):
        return BuildResult(sample, SampleStatus.FAILED, "CMake configure failed")

    if not run(["cmake", "--build", str(tmp_build_dir), "--config", configuration]):
        return BuildResult(sample, SampleStatus.FAILED, "CMake build failed")

    copied = copy_cmake_outputs(tmp_build_dir, target_build_dir, configuration)
    if copied == 0:
        return BuildResult(sample, SampleStatus.FAILED, "CMake build succeeded but no binary output was found")

    runtime_commands = [
        RuntimeCommand(sample, path.name, [str(path)], target_build_dir)
        for path in find_runnable_files(target_build_dir)
    ]
    return BuildResult(sample, SampleStatus.OK, copied_files=copied, runtime_commands=runtime_commands)


def build_dotnet_sample(sample: Sample, dotnet_tmp_root: Path, final_build_root: Path, configuration: str) -> BuildResult:
    tmp_build_dir = dotnet_tmp_root / sample.name
    target_build_dir = final_build_root / sample.name
    obj_dir = tmp_build_dir / "obj"

    if shutil.which("dotnet") is None:
        return BuildResult(sample, SampleStatus.MISSING_DEPENDENCY, "dotnet CLI not found")

    if not run([
        "dotnet", "build", str(sample.entry), "-c", configuration,
        "--output", str(tmp_build_dir), f"-p:BaseIntermediateOutputPath={obj_dir}{os.sep}",
    ]):
        return BuildResult(sample, SampleStatus.FAILED, "dotnet build failed")

    copied = copy_files(tmp_build_dir, target_build_dir)
    if copied == 0:
        return BuildResult(sample, SampleStatus.FAILED, "dotnet build succeeded but no output files were copied")

    command = find_dotnet_command(sample, target_build_dir)
    runtime_commands = [command] if command else []
    return BuildResult(sample, SampleStatus.OK, copied_files=copied, runtime_commands=runtime_commands)


def check_python_sample(sample: Sample, samples_root: Path, final_build_root: Path) -> BuildResult:
    del samples_root, final_build_root
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(sample.entry)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if compile_result.returncode != 0:
        print(compile_result.stdout, file=sys.stderr)
        return BuildResult(sample, SampleStatus.FAILED, "Python syntax check failed")

    runtime_command = RuntimeCommand(
        sample,
        sample.entry.name,
        [sys.executable, str(sample.entry)],
        sample.path,
    )
    return BuildResult(
        sample,
        SampleStatus.OK,
        "Python syntax check passed",
        runtime_commands=[runtime_command],
    )


def build_sample(
    sample: Sample,
    samples_root: Path,
    cmake_tmp_root: Path,
    dotnet_tmp_root: Path,
    final_build_root: Path,
    configuration: str,
) -> BuildResult:
    if sample.sample_type is SampleType.CMAKE:
        return build_cmake_sample(sample, cmake_tmp_root, final_build_root, configuration)
    if sample.sample_type is SampleType.DOTNET:
        return build_dotnet_sample(sample, dotnet_tmp_root, final_build_root, configuration)
    if sample.sample_type is SampleType.PYTHON:
        return check_python_sample(sample, samples_root, final_build_root)
    return BuildResult(sample, SampleStatus.FAILED, f"Unsupported sample type: {sample.sample_type}")


def print_available_samples(samples: list[Sample], root: Path) -> None:
    print("Available samples:")
    for sample in samples:
        rel = sample.path.relative_to(root)
        marker = " platform-specific" if sample.is_platform_specific else ""
        print(f"  - {sample.name} [{sample.sample_type.value}{marker}] ({rel})")


def print_summary(results: list[BuildResult]) -> None:
    counts = {status: 0 for status in SampleStatus}
    for result in results:
        counts[result.status] += 1

    print()
    print("==========================================")
    print("  Build summary")
    print("==========================================")
    print(f"  Samples selected       : {len(results)}")
    print(f"  Succeeded              : {counts[SampleStatus.OK]}")
    print(f"  Missing dependencies   : {counts[SampleStatus.MISSING_DEPENDENCY]}")
    print(f"  Skipped                : {counts[SampleStatus.SKIPPED]}")
    print(f"  Failed                 : {counts[SampleStatus.FAILED]}")

    notable = [result for result in results if result.status is not SampleStatus.OK]
    if notable:
        print()
        print("  Non-ok samples:")
        for result in notable:
            message = f": {result.message}" if result.message else ""
            print(f"    - {result.sample.name} [{result.status.value}]{message}")
    print("==========================================")
    print()


def print_runtime_summary(results: list[RuntimeResult]) -> None:
    counts = {status: 0 for status in RuntimeStatus}
    for result in results:
        counts[result.status] += 1

    print()
    print("==========================================")
    print("  Runtime attempt summary")
    print("==========================================")
    print(f"  Programs attempted      : {len(results)}")
    print(f"  Passed                  : {counts[RuntimeStatus.PASSED]}")
    print(f"  No camera/failed        : {counts[RuntimeStatus.NO_CAMERA_OR_FAILED]}")
    print(f"  Timed out               : {counts[RuntimeStatus.TIMED_OUT]}")
    print(f"  Could not start         : {counts[RuntimeStatus.COULD_NOT_START]}")

    notable = [result for result in results if result.status is not RuntimeStatus.PASSED]
    if notable:
        print()
        print("  Non-passing runtime attempts:")
        for result in notable:
            detail = result.message or f"exit code {result.exit_code}"
            print(f"    - {result.command.sample.name}: {result.status.value}: {detail}")
            print(f"::warning::{result.command.sample.name} runtime {result.status.value}: {detail}")
    print("==========================================")
    print()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--run-timeout-seconds must be positive") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--run-timeout-seconds must be positive")
    return parsed


def exit_code_for(results: list[BuildResult]) -> int:
    failing_statuses = {SampleStatus.FAILED, SampleStatus.MISSING_DEPENDENCY}
    return 1 if any(result.status in failing_statuses for result in results) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/check Application-Samples entries.")
    parser.add_argument("--all", action="store_true", help="Build/check all discovered samples without opening the interactive selector.")
    parser.add_argument("--sample", action="append", default=[], help="Sample name or path to build/check. Can be passed multiple times.")
    parser.add_argument("--type", choices=[item.value for item in SampleType], action="append", default=[], help="Sample type to build/check. Can be passed multiple times.")
    parser.add_argument("--skip-platform-specific", action="store_true", help="Skip samples that are not under a cross-platform/ folder.")
    parser.add_argument("--run", action="store_true", help="Attempt runnable outputs after successful builds/checks.")
    parser.add_argument("--run-timeout-seconds", type=positive_int, default=90, help="Timeout for each runtime attempt. Default: 90.")
    parser.add_argument("--clean", action="store_true", help="Delete build and temporary directories before running.")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete root .cmake-tmp/.dotnet-tmp directories after building.")
    parser.add_argument("--configuration", default="Release", help="Build configuration for CMake and dotnet samples. Default: Release.")
    parser.add_argument("--samples-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--skip-cmake", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-dotnet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-python", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 1

    root = repo_root()
    samples_root = args.samples_root.resolve() if args.samples_root else find_samples_root(root)
    output_root = args.output_root.resolve() if args.output_root else root
    final_build_root = output_root / "build"
    cmake_tmp_root = output_root / ".cmake-tmp"
    dotnet_tmp_root = output_root / ".dotnet-tmp"

    samples = discover_samples(samples_root)
    if not samples:
        print(f"No samples found under {samples_root}", file=sys.stderr)
        return 1

    selected_samples, missing_selectors = select_samples(samples, samples_root, args)
    if missing_selectors:
        for selector in missing_selectors:
            print(f"Unknown sample selector: {selector}", file=sys.stderr)
        print_available_samples(samples, samples_root)
        return 1

    if not (args.all or args.sample):
        selected_samples = choose_samples_tui(selected_samples, samples_root)
        if not selected_samples:
            print("No samples selected.")
            return 0

    if not selected_samples:
        print("No samples selected.", file=sys.stderr)
        print_available_samples(samples, samples_root)
        return 1

    if args.clean:
        remove_tree(cmake_tmp_root)
        remove_tree(dotnet_tmp_root)
        remove_tree(final_build_root)

    results: list[BuildResult] = []
    try:
        for sample in selected_samples:
            print("==========================================")
            print(f"{sample.sample_type.value}: {sample.name}")
            print("==========================================")

            if args.skip_platform_specific and sample.is_platform_specific:
                results.append(BuildResult(
                    sample,
                    SampleStatus.SKIPPED,
                    "platform-specific sample skipped by --skip-platform-specific",
                ))
                remove_sample_work_dirs(sample.path)
                continue

            remove_tree(final_build_root / sample.name)
            result = build_sample(
                sample,
                samples_root,
                cmake_tmp_root,
                dotnet_tmp_root,
                final_build_root,
                args.configuration,
            )
            remove_sample_work_dirs(sample.path)
            results.append(result)
            status_line = result.status.value
            if result.message:
                status_line += f": {result.message}"
            if result.copied_files:
                status_line += f" ({result.copied_files} file(s))"
            print(status_line)
    finally:
        if not args.keep_temp:
            remove_tree(cmake_tmp_root)
            remove_tree(dotnet_tmp_root)

    print_summary(results)
    build_exit_code = exit_code_for(results)
    if args.run and build_exit_code == 0:
        runtime_commands = [
            command
            for result in results
            for command in runtime_commands_for_build_result(result)
        ]
        if runtime_commands:
            runtime_results = run_runtime_commands(runtime_commands, args.run_timeout_seconds)
            print_runtime_summary(runtime_results)
    return build_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
