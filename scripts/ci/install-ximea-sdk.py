#!/usr/bin/env python3
"""Install the latest XIMEA SDK beta on GitHub-hosted CI runners."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LINUX_X64_URL = "https://www.ximea.com/getattachment/ab5baacf-e806-4b9d-b3d4-7eedf0f092b8/XIMEA_Linux_SP.tgz"
LINUX_ARM64_URL = "https://www.ximea.com/getattachment/21790810-a3ed-4183-948f-e999cfcf8233/XIMEA_Linux_ARM_SP.tgz"
MACOS_X64_URL = "https://www.ximea.com/getattachment/1cbfaa8e-a175-4dab-badd-ab2961387799/XIMEA_macOX_SP.dmg"
MACOS_ARM64_URL = "https://www.ximea.com/getattachment/8e005503-9914-4208-a80b-509dbbb3a901/XIMEA_macOS_ARM_SP.dmg"
WINDOWS_URL = "https://www.ximea.com/getattachment/23c0b9e6-5c24-4d27-9a6e-b377d9390c3d/XIMEA_Windows_SP_Beta.exe"

PLATFORMS = {"linux", "macos", "windows"}


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set by GitHub Actions")
    return value


def run_checked(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def download(url: str, destination: Path, *, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} to {destination} (attempt {attempt}/{attempts})", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Application-Samples-CI"})
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
            return
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            last_error = error
            if destination.exists():
                destination.unlink()
            print(f"Download attempt {attempt} failed: {error}", file=sys.stderr, flush=True)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def extract_tgz(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def copy_headers(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for header in source_dir.glob("*.h"):
        shutil.copy2(header, target_dir / header.name)


def copy_xiapiplus_support(package_root: Path, sdk_root: Path) -> None:
    source = package_root / "samples" / "_libs" / "xiAPIplus"
    shutil.copytree(source, sdk_root / "xiAPIplus", dirs_exist_ok=True)
    include = sdk_root / "include"
    files = {
        source / "xiapiplus.h": include / "xiApiPlus.h",
        source / "xiAPIplus_core.cpp": include / "xiAPIplus_core.cpp",
        source / "xiAPIplus_parameters.cpp": include / "xiAPIplus_parameters.cpp",
        source / "xiAPIplus_tiff.cpp": include / "xiAPIplus_tiff.cpp",
        source / "xiAPIplus_tiff.h": include / "xiAPIplus_tiff.h",
        package_root / "samples" / "_libs" / "os_common_header.h": sdk_root / "os_common_header.h",
    }
    for src, dst in files.items():
        shutil.copy2(src, dst)


def append_github_env(values: dict[str, str]) -> None:
    env_file = Path(require_env("GITHUB_ENV"))
    with env_file.open("a", encoding="utf-8") as handle:
        for name, value in values.items():
            handle.write(f"{name}={value}\n")


def append_github_path(path: Path) -> None:
    path_file = Path(require_env("GITHUB_PATH"))
    with path_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{path}\n")


def linux_config() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine == "x86_64":
        return LINUX_X64_URL, "X64"
    if machine in {"aarch64", "arm64"}:
        return LINUX_ARM64_URL, "Xarm64"
    raise RuntimeError(f"Unsupported Linux architecture: {platform.machine()}")


def install_linux() -> None:
    sdk_url, sdk_arch = linux_config()
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    extract_dir = runner_temp / "ximea-package"
    sdk_root = runner_temp / "ximea-sdk"
    machine = platform.machine()
    archive = download_dir / f"XIMEA_Linux_SP_latest_{machine}.tgz"

    (sdk_root / "include" / "m3api").mkdir(parents=True, exist_ok=True)
    (sdk_root / "lib").mkdir(parents=True, exist_ok=True)
    download(sdk_url, archive)
    extract_tgz(archive, extract_dir)

    package_root = extract_dir / "package"
    version_file = package_root / "version_LINUX_SP.txt"
    if version_file.is_file():
        print(f"Downloaded XIMEA Linux SDK {version_file.read_text(encoding='utf-8').strip()}")

    run_checked(["sudo", "apt-get", "update"])
    run_checked(["sudo", "apt-get", "install", "--yes", "libraw1394-11", "libtiff6", "libusb-1.0-0"])
    run_checked(["sudo", "install", "-m", "0644", str(package_root / "api" / sdk_arch / "libm3api.so.2"), "/usr/lib/libm3api.so.2"])
    run_checked(["sudo", "ln", "-sfn", "libm3api.so.2", "/usr/lib/libm3api.so"])

    ldconfig = run_checked(["ldconfig", "-p"], stdout=subprocess.PIPE)
    libtiff6 = next((line.rsplit(" => ", 1)[1].strip() for line in ldconfig.stdout.splitlines() if "libtiff.so.6" in line and " => " in line), "")
    if not libtiff6:
        raise RuntimeError("Could not locate libtiff.so.6 with ldconfig")
    run_checked(["sudo", "ln", "-sfn", libtiff6, "/usr/lib/libtiff.so.5"])
    run_checked(["sudo", "ldconfig"])

    copy_headers(package_root / "include", sdk_root / "include")
    copy_headers(package_root / "include", sdk_root / "include" / "m3api")
    copy_xiapiplus_support(package_root, sdk_root)
    shutil.copy2(package_root / "api" / sdk_arch / "libm3api.so.2", sdk_root / "lib" / "libm3api.so.2")
    sdk_library_link = sdk_root / "lib" / "libm3api.so"
    if sdk_library_link.exists() or sdk_library_link.is_symlink():
        sdk_library_link.unlink()
    sdk_library_link.symlink_to("libm3api.so.2")

    values = {
        "XIMEA_ROOT": str(sdk_root),
        "XIMEA_SP_PATH": str(sdk_root),
        "PYTHONPATH": f"{package_root / 'api' / 'Python' / 'v3'}{os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''}",
        "LD_LIBRARY_PATH": f"{sdk_root / 'lib'}{os.pathsep + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''}",
    }
    os.environ.update(values)
    append_github_env(values)
    print(f"Installed latest XIMEA SDK beta for Linux {machine} at {sdk_root}")


def macos_config() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine == "x86_64":
        return MACOS_X64_URL, "x86_64"
    if machine == "arm64":
        return MACOS_ARM64_URL, "arm64"
    raise RuntimeError(f"Unsupported macOS architecture: {platform.machine()}")


def install_macos() -> None:
    sdk_url, expected_arch = macos_config()
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    support_dir = runner_temp / "ximea-portable-sources"
    python_root = runner_temp / "ximea-python"
    sdk_root = runner_temp / "ximea-sdk"
    mount_dir = runner_temp / "ximea-volume"
    machine = platform.machine()
    dmg = download_dir / f"XIMEA_macOS_SP_latest_{machine}.dmg"
    support_archive = download_dir / "XIMEA_Linux_SP_latest.tgz"

    for path in (download_dir, support_dir, python_root, sdk_root / "include" / "m3api", mount_dir):
        path.mkdir(parents=True, exist_ok=True)
    download(sdk_url, dmg)
    download(LINUX_X64_URL, support_archive)

    attached = False
    try:
        run_checked(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount_dir), str(dmg)])
        attached = True
        if not (mount_dir / "m3api.framework").is_dir():
            raise FileNotFoundError(f"Missing m3api framework: {mount_dir / 'm3api.framework'}")
        if not (mount_dir / "Examples" / "xiPython" / "v3" / "ximea").is_dir():
            raise FileNotFoundError(f"Missing ximea Python package: {mount_dir / 'Examples' / 'xiPython' / 'v3' / 'ximea'}")
        run_checked(["sudo", "rm", "-rf", "/Library/Frameworks/m3api.framework"])
        run_checked(["sudo", "ditto", str(mount_dir / "m3api.framework"), "/Library/Frameworks/m3api.framework"])
        run_checked(["sudo", "xattr", "-dr", "com.apple.quarantine", "/Library/Frameworks/m3api.framework"])
        run_checked(["codesign", "--verify", "--deep", "--strict", "/Library/Frameworks/m3api.framework"])
        shutil.copytree(mount_dir / "Examples" / "xiPython" / "v3" / "ximea", python_root / "ximea", dirs_exist_ok=True)
    finally:
        if attached:
            subprocess.run(["hdiutil", "detach", str(mount_dir)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    extract_tgz(support_archive, support_dir)
    portable_root = support_dir / "package"
    version_file = portable_root / "version_LINUX_SP.txt"
    if version_file.is_file():
        print(f"Downloaded XIMEA Linux portable sources {version_file.read_text(encoding='utf-8').strip()}")

    copy_headers(portable_root / "include", sdk_root / "include")
    copy_headers(portable_root / "include", sdk_root / "include" / "m3api")
    framework_header = Path("/Library/Frameworks/m3api.framework/Headers/xiApi.h")
    shutil.copy2(framework_header, sdk_root / "include" / "xiApi.h")
    shutil.copy2(framework_header, sdk_root / "include" / "m3api" / "xiApi.h")
    copy_xiapiplus_support(portable_root, sdk_root)

    lipo = run_checked(["lipo", "-archs", "/Library/Frameworks/m3api.framework/m3api"], stdout=subprocess.PIPE)
    arches = set(lipo.stdout.split())
    if expected_arch not in arches:
        raise RuntimeError(f"m3api framework does not contain {expected_arch}: {lipo.stdout.strip()}")

    values = {
        "XIMEA_ROOT": str(sdk_root),
        "XIMEA_SP_PATH": str(sdk_root),
        "PYTHONPATH": f"{python_root}{os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''}",
        "DYLD_FRAMEWORK_PATH": f"/Library/Frameworks{os.pathsep + os.environ['DYLD_FRAMEWORK_PATH'] if os.environ.get('DYLD_FRAMEWORK_PATH') else ''}",
    }
    os.environ.update(values)
    append_github_env(values)
    print(f"Installed latest XIMEA SDK beta for macOS {machine} at {sdk_root}")


def validate_windows_signature(installer: Path) -> None:
    script = (
        "param([string]$Installer); "
        "Import-Module Microsoft.PowerShell.Security -ErrorAction Stop; "
        "$signature = Get-AuthenticodeSignature -FilePath $Installer; "
        "if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'XIMEA') { "
        "  throw 'XIMEA SDK Authenticode signature is not valid' "
        "}"
    )
    run_checked(power_shell_command(script, str(installer)))


def install_windows() -> None:
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    installer = download_dir / "XIMEA_Windows_SP_Beta_latest.exe"

    download(WINDOWS_URL, installer)
    validate_windows_signature(installer)
    completed = run_checked([str(installer), "/S"])
    if completed.returncode != 0:
        raise RuntimeError(f"XIMEA SDK installer failed with exit code {completed.returncode}")

    ximea_sp_path = os.environ.get("XIMEA_SP_PATH") or read_windows_environment("XIMEA_SP_PATH")
    if not ximea_sp_path:
        raise RuntimeError("XIMEA SDK installer completed but did not set XIMEA_SP_PATH")

    sdk_root = Path(ximea_sp_path)
    values = {
        "XIMEA_SP_PATH": str(sdk_root),
        "PYTHONPATH": str(sdk_root / "API" / "Python" / "v3"),
    }
    os.environ.update(values)
    os.environ["PATH"] = f"{sdk_root / 'API' / 'xiAPI'}{os.pathsep}{os.environ.get('PATH', '')}"
    append_github_env(values)
    append_github_path(sdk_root / "API" / "xiAPI")
    print(f"Installed latest XIMEA SDK beta for Windows x64 at {sdk_root}")


def read_windows_environment(name: str) -> str:
    script = (
        "param([string]$Name); "
        "$value = [Environment]::GetEnvironmentVariable($Name, 'Machine'); "
        "if ([string]::IsNullOrWhiteSpace($value)) { $value = [Environment]::GetEnvironmentVariable($Name, 'User') }; "
        "if ($value) { Write-Output $value }"
    )
    result = run_checked(power_shell_command(script, name), stdout=subprocess.PIPE)
    return result.stdout.strip()


def power_shell_command(script: str, *args: str) -> list[str]:
    executable = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    return [executable, "-NoProfile", "-NonInteractive", "-Command", script, *args]


def detect_platform() -> str:
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    raise RuntimeError(f"Unsupported operating system: {platform.system()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install the latest XIMEA SDK beta for GitHub-hosted CI.")
    parser.add_argument("--platform", choices=sorted(PLATFORMS), help="Override platform auto-detection.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    selected_platform = args.platform or detect_platform()
    try:
        if selected_platform == "linux":
            install_linux()
        elif selected_platform == "macos":
            install_macos()
        elif selected_platform == "windows":
            install_windows()
        else:
            parser.error(f"unsupported platform: {selected_platform}")
        return 0
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        print(f"XIMEA SDK installation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
