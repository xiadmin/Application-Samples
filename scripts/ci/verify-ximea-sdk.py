#!/usr/bin/env python3
"""Verify concrete XIMEA SDK files staged for GitHub-hosted CI."""

from __future__ import annotations

import argparse
import os
import platform as host_platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PLATFORMS = {"linux-x64", "linux-arm64", "windows-x64", "macos-x64", "macos-arm64"}
EXPECTED_VERSION_PREFIX = "4.33.21"


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def run_checked(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def verify_python_import(python: str) -> None:
    result = run_checked([
        python,
        "-c",
        "import ximea; from ximea import xiapi; print(ximea.__version__); print(xiapi.__file__)",
    ])
    print(result.stdout, end="")
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    if not first_line.startswith(EXPECTED_VERSION_PREFIX):
        raise RuntimeError(f"Unexpected ximea Python version: {first_line}")


def verify_cmake_probe(repo_root: Path, sdk_root: Path, cmake: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ximea-cmake-probe-") as tmp_text:
        tmp = Path(tmp_text)
        source = tmp / "src"
        build = tmp / "build"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(ximea_probe LANGUAGES C CXX)\n"
            f"include(\"{(repo_root / 'cmake' / 'CMakeLists.txt').as_posix()}\")\n"
            "ximea_find_sdk(REQUIRED)\n"
            "if(NOT TARGET XIMEA::xiAPI)\n"
            "  message(FATAL_ERROR \"XIMEA::xiAPI target missing\")\n"
            "endif()\n"
            "if(NOT TARGET XIMEA::xiAPIplus)\n"
            "  message(FATAL_ERROR \"XIMEA::xiAPIplus target missing\")\n"
            "endif()\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["XIMEA_ROOT"] = str(sdk_root)
        run_checked([cmake, "-S", str(source), "-B", str(build)], env=env)


def verify_linux(sdk_root: Path, repo_root: Path, python: str, cmake: str) -> None:
    require_file(sdk_root / "include" / "xiApi.h", "xiAPI header")
    require_file(sdk_root / "include" / "wintypedefs.h", "xiAPI typedef compatibility header")
    require_file(sdk_root / "include" / "m3Identify.h", "xiAPI identification header")
    require_file(sdk_root / "include" / "m3api" / "wintypedefs.h", "m3api typedef compatibility header")
    require_file(sdk_root / "include" / "m3api" / "m3Identify.h", "m3api identification header")
    require_file(sdk_root / "include" / "xiApiPlus.h", "xiAPIplus header")
    require_file(sdk_root / "include" / "xiAPIplus_core.cpp", "xiAPIplus portable source")
    require_file(sdk_root / "os_common_header.h", "xiAPIplus OS compatibility header")
    require_file(sdk_root / "lib" / "libm3api.so.2", "xiAPI shared library")
    require_file(Path("/usr/lib/libm3api.so.2"), "system xiAPI shared library")
    verify_cmake_probe(repo_root, sdk_root, cmake)
    verify_python_import(python)


def verify_windows(sdk_root: Path, python: str) -> None:
    require_dir(sdk_root, "XIMEA_SP_PATH")
    require_file(sdk_root / "API" / "xiAPI" / "xiApi.h", "xiAPI header")
    require_file(sdk_root / "API" / "xiAPI" / "xiapi64.lib", "x64 xiAPI import library")
    require_file(sdk_root / "API" / "xiAPI" / "xiapi64.dll", "x64 xiAPI runtime DLL")
    require_file(sdk_root / "Examples" / "Sources" / "_libs" / "xiAPIplus" / "xiapiplus.h", "xiAPIplus header")
    require_file(sdk_root / "API" / "xiAPI.NET.NET.7.0" / "xiApi.NETX64.dll", "xiAPI.NET x64 assembly")
    require_file(sdk_root / "API" / "Python" / "v3" / "ximea" / "xiapi.py", "ximea Python package")
    verify_python_import(python)


def verify_macos(sdk_root: Path, framework_root: Path, repo_root: Path, python: str, cmake: str, expected_arch: str) -> None:
    framework = framework_root / "m3api.framework"
    binary = framework / "m3api"
    require_file(framework / "Headers" / "xiApi.h", "framework xiAPI header")
    require_file(binary, "m3api framework binary")
    require_file(sdk_root / "include" / "wintypedefs.h", "xiAPI typedef compatibility header")
    require_file(sdk_root / "include" / "m3api" / "wintypedefs.h", "m3api typedef compatibility header")
    require_file(sdk_root / "include" / "xiApiPlus.h", "normalized xiAPIplus header")
    require_file(sdk_root / "include" / "xiAPIplus_core.cpp", "normalized xiAPIplus source")
    require_file(sdk_root / "os_common_header.h", "xiAPIplus OS compatibility header")
    run_checked(["codesign", "--verify", "--deep", "--strict", str(framework)])
    lipo = run_checked(["lipo", "-archs", str(binary)])
    arches = set(lipo.stdout.split())
    if expected_arch not in arches:
        raise RuntimeError(f"m3api framework does not contain {expected_arch}: {lipo.stdout.strip()}")
    verify_cmake_probe(repo_root, sdk_root, cmake)
    verify_python_import(python)


def default_sdk_root(platform_name: str) -> Path:
    if platform_name == "windows-x64":
        return Path(os.environ.get("XIMEA_SP_PATH", ""))
    return Path(os.environ.get("XIMEA_ROOT", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify XIMEA SDK files for GitHub-hosted CI.")
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    parser.add_argument("--sdk-root", type=Path, help="Override XIMEA_ROOT/XIMEA_SP_PATH for tests or diagnostics.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--framework-root", type=Path, default=Path("/Library/Frameworks"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cmake", default=shutil.which("cmake") or "cmake")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    sdk_root = (args.sdk_root or default_sdk_root(args.platform)).resolve()
    try:
        print(f"Verifying XIMEA SDK for {args.platform}")
        print(f"SDK root: {sdk_root}")
        if args.platform == "linux-x64" or args.platform == "linux-arm64":
            verify_linux(sdk_root, args.repo_root.resolve(), args.python, args.cmake)
        elif args.platform == "windows-x64":
            verify_windows(sdk_root, args.python)
        elif args.platform == "macos-x64":
            verify_macos(sdk_root, args.framework_root.resolve(), args.repo_root.resolve(), args.python, args.cmake, "x86_64")
        elif args.platform == "macos-arm64":
            verify_macos(sdk_root, args.framework_root.resolve(), args.repo_root.resolve(), args.python, args.cmake, "arm64")
        else:
            parser.error(f"unsupported platform: {args.platform}")
        print("XIMEA SDK verification passed")
        return 0
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"XIMEA SDK verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
