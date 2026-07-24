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

LINUX_X64_URL = "https://updates.ximea.com/public/ximea_linux_sp_beta.tgz"
LINUX_ARM64_URL = "https://updates.ximea.com/public/ximea_linux_arm_sp_beta.tgz"
MACOS_X64_URL = "https://www.ximea.com/getattachment/1cbfaa8e-a175-4dab-badd-ab2961387799/XIMEA_macOX_SP.dmg"
MACOS_ARM64_URL = "https://www.ximea.com/getattachment/8e005503-9914-4208-a80b-509dbbb3a901/XIMEA_macOS_ARM_SP.dmg"
WINDOWS_URL = "https://www.ximea.com/getattachment/23c0b9e6-5c24-4d27-9a6e-b377d9390c3d/XIMEA_Windows_SP_Beta.exe"

LINUX_INSTALL_ROOT = Path("/opt/XIMEA")
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


def append_github_env(values: dict[str, str]) -> None:
    env_file = Path(require_env("GITHUB_ENV"))
    with env_file.open("a", encoding="utf-8") as handle:
        for name, value in values.items():
            handle.write(f"{name}={value}\n")


def append_github_path(path: Path) -> None:
    path_file = Path(require_env("GITHUB_PATH"))
    with path_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{path}\n")


def prepend_env(name: str, path: Path) -> str:
    current = os.environ.get(name)
    return f"{path}{os.pathsep + current if current else ''}"


def linux_config() -> str:
    machine = platform.machine().lower()
    if machine == "x86_64":
        return LINUX_X64_URL
    if machine in {"aarch64", "arm64"}:
        return LINUX_ARM64_URL
    raise RuntimeError(f"Unsupported Linux architecture: {platform.machine()}")


def install_linux() -> None:
    sdk_url = linux_config()
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    extract_dir = runner_temp / "ximea-package"
    machine = platform.machine()
    archive = download_dir / f"XIMEA_Linux_SP_beta_{machine}.tgz"

    download(sdk_url, archive)
    extract_tgz(archive, extract_dir)
    package_root = extract_dir / "package"
    installer = package_root / "install"
    if not installer.is_file():
        raise FileNotFoundError(f"Missing XIMEA Linux installer script: {installer}")

    version_file = package_root / "version_LINUX_SP.txt"
    if version_file.is_file():
        print(f"Downloaded XIMEA Linux SDK {version_file.read_text(encoding='utf-8').strip()}")

    run_checked(["sudo", "apt-get", "update"])
    run_checked([
        "sudo",
        "apt-get",
        "install",
        "--yes",
        "build-essential",
        "cmake",
        "g++",
        "gcc",
        "libraw1394-11",
        "libtiff5",
        "libusb-1.0-0",
    ])

    run_checked(["./install"], cwd=package_root)

    values = {
        "XIMEA_ROOT": str(LINUX_INSTALL_ROOT),
        "XIMEA_SP_PATH": str(LINUX_INSTALL_ROOT),
        "PYTHONPATH": prepend_env("PYTHONPATH", package_root / "api" / "Python" / "v3"),
        "LD_LIBRARY_PATH": prepend_env("LD_LIBRARY_PATH", LINUX_INSTALL_ROOT / "lib"),
    }
    os.environ.update(values)
    append_github_env(values)
    print(f"Installed latest XIMEA SDK beta for Linux {machine} with XIMEA's installer script")


def macos_config() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine == "x86_64":
        return MACOS_X64_URL, "x86_64"
    if machine == "arm64":
        return MACOS_ARM64_URL, "arm64"
    raise RuntimeError(f"Unsupported macOS architecture: {platform.machine()}")


def find_macos_install_script(mount_dir: Path) -> Path:
    candidates = [mount_dir / "install"]
    candidates.extend(path for path in mount_dir.rglob("install") if path.is_file())
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    mounted_entries = ", ".join(sorted(path.relative_to(mount_dir).as_posix() for path in mount_dir.iterdir()))
    raise FileNotFoundError(
        "Missing XIMEA macOS install script on mounted XIMEA volume"
        f" {mount_dir}; top-level entries: {mounted_entries or '<empty>'}"
    )


def find_macos_sdk_root() -> Path:
    candidates = (
        Path("/Applications/XIMEA"),
        Path("/Library/XIMEA"),
        Path("/Library/Application Support/XIMEA"),
        Path("/opt/XIMEA"),
    )
    for candidate in candidates:
        if (candidate / "Examples" / "Sources" / "_libs" / "xiAPIplus" / "xiapiplus.h").is_file():
            return candidate

    roots = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate XIMEA macOS SDK examples after installer completed; checked {roots}")


def install_macos() -> None:
    sdk_url, expected_arch = macos_config()
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    mount_dir = runner_temp / "ximea-volume"
    machine = platform.machine()
    dmg = download_dir / f"XIMEA_macOS_SP_beta_{machine}.dmg"

    mount_dir.mkdir(parents=True, exist_ok=True)
    download(sdk_url, dmg)

    attached = False
    try:
        run_checked(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount_dir), str(dmg)])
        attached = True
        installer = find_macos_install_script(mount_dir)
        run_checked(["chmod", "+x", str(installer)])
        run_checked([str(installer)])
    finally:
        if attached:
            subprocess.run(
                ["hdiutil", "detach", str(mount_dir)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    sdk_root = find_macos_sdk_root()
    values = {
        "XIMEA_ROOT": str(sdk_root),
        "XIMEA_SP_PATH": str(sdk_root),
        "DYLD_FRAMEWORK_PATH": prepend_env("DYLD_FRAMEWORK_PATH", Path("/Library/Frameworks")),
    }
    os.environ.update(values)
    append_github_env(values)

    lipo = run_checked(["lipo", "-archs", "/Library/Frameworks/m3api.framework/m3api"], stdout=subprocess.PIPE)
    arches = set(lipo.stdout.split())
    if expected_arch not in arches:
        raise RuntimeError(f"m3api framework does not contain {expected_arch}: {lipo.stdout.strip()}")
    print(f"Installed latest XIMEA SDK beta for macOS {machine} with XIMEA's installer script")


def validate_windows_signature(installer: Path) -> None:
    installer_literal = power_shell_literal(str(installer))
    script = (
        f"$Installer = {installer_literal}; "
        "Import-Module Microsoft.PowerShell.Security -ErrorAction Stop; "
        "$signature = Get-AuthenticodeSignature -FilePath $Installer; "
        "if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'XIMEA') { "
        "  throw 'XIMEA SDK Authenticode signature is not valid' "
        "}"
    )
    run_checked(power_shell_command(script))


def install_windows() -> None:
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    installer = download_dir / "XIMEA_Windows_SP_Beta_latest.exe"

    download(WINDOWS_URL, installer)
    validate_windows_signature(installer)
    run_checked([str(installer), "/S"])

    ximea_sp_path = os.environ.get("XIMEA_SP_PATH") or read_windows_environment("XIMEA_SP_PATH")
    if not ximea_sp_path:
        raise RuntimeError("XIMEA SDK installer completed but did not set XIMEA_SP_PATH")

    sdk_root = Path(ximea_sp_path)
    values = {
        "PYTHONPATH": str(sdk_root / "API" / "Python" / "v3"),
    }
    os.environ.update(values)
    os.environ["PATH"] = f"{sdk_root / 'API' / 'xiAPI'}{os.pathsep}{os.environ.get('PATH', '')}"
    append_github_env(values)
    append_github_path(sdk_root / "API" / "xiAPI")
    print(f"Installed latest XIMEA SDK beta for Windows x64 with XIMEA's installer")


def read_windows_environment(name: str) -> str:
    name_literal = power_shell_literal(name)
    script = (
        f"$Name = {name_literal}; "
        "$value = [Environment]::GetEnvironmentVariable($Name, 'Machine'); "
        "if ([string]::IsNullOrWhiteSpace($value)) { $value = [Environment]::GetEnvironmentVariable($Name, 'User') }; "
        "if ($value) { Write-Output $value }"
    )
    result = run_checked(power_shell_command(script), stdout=subprocess.PIPE)
    return result.stdout.strip()


def power_shell_command(script: str) -> list[str]:
    executable = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    return [executable, "-NoProfile", "-NonInteractive", "-Command", script]


def power_shell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
