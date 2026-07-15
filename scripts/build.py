#!/usr/bin/env python3
"""Build all samples."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import find_samples_root, folder_name_for, repo_root


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run(cmd: list[str]) -> bool:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


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


def copy_cmake_outputs(tmp_build_dir: Path, target_build_dir: Path) -> None:
    built_bin_dir = tmp_build_dir / "build"
    built_release_dir = built_bin_dir / "Release"

    if built_release_dir.is_dir():
        copy_files(built_release_dir, target_build_dir)
        return
    if built_bin_dir.is_dir():
        copy_files(built_bin_dir, target_build_dir)
        return

    # Fallback matching the PowerShell script: copy .exe files and extensionless
    # binaries, avoiding obvious CMake metadata.
    target_build_dir.mkdir(parents=True, exist_ok=True)
    for item in tmp_build_dir.rglob("*"):
        if not item.is_file():
            continue
        as_text = str(item)
        if "CMakeFiles" in as_text or item.name == "Makefile":
            continue
        if item.suffix == ".exe" or item.suffix == "":
            shutil.copy2(item, target_build_dir / item.name)


def build_cmake_samples(samples_root: Path, cmake_tmp_root: Path, final_build_root: Path, failed: list[str]) -> tuple[int, int]:
    found = ok = 0
    print(f"Finding C and C++ samples in {samples_root}...")
    for cmake_file in sorted(samples_root.rglob("CMakeLists.txt")):
        sample_dir = cmake_file.parent
        if sample_dir.name not in {"c", "cpp"}:
            continue

        found += 1
        folder_name = folder_name_for(samples_root, sample_dir)
        print("==========================================")
        print(f"Building: {folder_name}")
        print("==========================================")

        tmp_build_dir = cmake_tmp_root / folder_name
        target_build_dir = final_build_root / folder_name

        if not run(["cmake", "-S", str(sample_dir), "-B", str(tmp_build_dir)]):
            print(f"WARNING: CMake configure failed for {folder_name}", file=sys.stderr)
            failed.append(folder_name)
            continue

        if not run(["cmake", "--build", str(tmp_build_dir), "--config", "Release"]):
            print(f"WARNING: CMake build failed for {folder_name}", file=sys.stderr)
            failed.append(folder_name)
            continue

        copy_cmake_outputs(tmp_build_dir, target_build_dir)
        ok += 1

    return found, ok


def build_dotnet_samples(samples_root: Path, dotnet_tmp_root: Path, final_build_root: Path, failed: list[str]) -> tuple[int, int]:
    found = ok = 0
    print()
    print(f"Finding C# samples in {samples_root}...")
    for csproj_file in sorted(samples_root.rglob("*.csproj")):
        sample_dir = csproj_file.parent
        found += 1
        folder_name = folder_name_for(samples_root, sample_dir)
        print("==========================================")
        print(f"Building: {folder_name}")
        print("==========================================")

        tmp_build_dir = dotnet_tmp_root / folder_name
        target_build_dir = final_build_root / folder_name
        obj_dir = tmp_build_dir / "obj"

        if not run([
            "dotnet", "build", str(csproj_file), "-c", "Release",
            "--output", str(tmp_build_dir), f"-p:BaseIntermediateOutputPath={obj_dir}{os.sep}",
        ]):
            print(f"WARNING: dotnet build failed for {folder_name}", file=sys.stderr)
            failed.append(folder_name)
            continue

        copy_files(tmp_build_dir, target_build_dir)
        ok += 1

    return found, ok


def check_python_samples(samples_root: Path, final_build_root: Path, failed: list[str]) -> tuple[int, int]:
    found = ok = 0
    print()
    print(f"Finding Python samples in {samples_root}...")
    has_ximea = importlib.util.find_spec("ximea") is not None

    for py_main in sorted(samples_root.rglob("main.py")):
        sample_dir = py_main.parent
        found += 1
        folder_name = folder_name_for(samples_root, sample_dir)
        relative_path = sample_dir.relative_to(samples_root)
        print("==========================================")
        print(f"Checking Python sample: {folder_name}")
        print("==========================================")

        if not has_ximea:
            print(
                f"WARNING: ximea module not found for {folder_name} -- "
                "install the XIMEA SDK to get site-packages/ximea",
                file=sys.stderr,
            )
            failed.append(folder_name)
            continue

        target_build_dir = final_build_root / folder_name
        target_build_dir.mkdir(parents=True, exist_ok=True)

        # Keep a PowerShell launcher for compatibility with the existing build layout,
        # and add a cross-platform Python launcher.
        ps_rel = "\\".join(("..", "..", samples_root.name, *relative_path.parts, "main.py"))
        (target_build_dir / "run.ps1").write_text(
            f'& python "$PSScriptRoot\\{ps_rel}" @args\n', encoding="utf-8"
        )

        py_rel = "/".join(("..", "..", samples_root.name, *relative_path.parts, "main.py"))
        (target_build_dir / "run.py").write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "import runpy\n"
            "import sys\n\n"
            f"target = Path(__file__).resolve().parent / {py_rel!r}\n"
            "sys.argv = [str(target), *sys.argv[1:]]\n"
            "runpy.run_path(str(target), run_name='__main__')\n",
            encoding="utf-8",
        )
        ok += 1

    return found, ok


def print_summary(total_found: int, total_ok: int, failed: list[str]) -> None:
    print()
    print("==========================================")
    print("  Build summary")
    print("==========================================")
    print(f"  Samples found  : {total_found}")
    print(f"  Succeeded      : {total_ok}")
    print(f"  Failed         : {len(failed)}")
    if failed:
        print()
        print("  Failed samples:")
        for name in failed:
            print(f"    - {name}")
    print("==========================================")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all Application-Samples entries.")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete .cmake-tmp/.dotnet-tmp after building.")
    parser.add_argument("--skip-cmake", action="store_true", help="Skip C/C++ CMake samples.")
    parser.add_argument("--skip-dotnet", action="store_true", help="Skip C# .NET samples.")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python launcher generation.")
    args = parser.parse_args()

    root = repo_root()
    samples_root = find_samples_root(root)
    final_build_root = root / "build"
    cmake_tmp_root = root / ".cmake-tmp"
    dotnet_tmp_root = root / ".dotnet-tmp"

    remove_tree(cmake_tmp_root)
    remove_tree(dotnet_tmp_root)
    remove_tree(final_build_root)

    total_found = total_ok = 0
    failed: list[str] = []

    try:
        if not args.skip_cmake:
            found, ok = build_cmake_samples(samples_root, cmake_tmp_root, final_build_root, failed)
            total_found += found
            total_ok += ok
            print("==========================================")
            print("Cleaning up CMake temporary files...")
            if not args.keep_temp:
                remove_tree(cmake_tmp_root)

        if not args.skip_dotnet:
            found, ok = build_dotnet_samples(samples_root, dotnet_tmp_root, final_build_root, failed)
            total_found += found
            total_ok += ok
            print("==========================================")
            print("Cleaning up dotnet temporary files...")
            if not args.keep_temp:
                remove_tree(dotnet_tmp_root)

        if not args.skip_python:
            found, ok = check_python_samples(samples_root, final_build_root, failed)
            total_found += found
            total_ok += ok
    finally:
        if not args.keep_temp:
            remove_tree(cmake_tmp_root)
            remove_tree(dotnet_tmp_root)

    print_summary(total_found, total_ok, failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
