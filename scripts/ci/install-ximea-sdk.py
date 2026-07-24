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


def linux_config() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine == "x86_64":
        return LINUX_X64_URL, "x86_64"
    if machine in {"aarch64", "arm64"}:
        return LINUX_ARM64_URL, "arm64"
    raise RuntimeError(f"Unsupported Linux architecture: {platform.machine()}")


def install_linux(pcie: bool = False) -> None:
    sdk_url, _arch = linux_config()
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    extract_dir = runner_temp / "ximea-package"
    machine = platform.machine()
    archive = download_dir / f"XIMEA_Linux_SP_beta_{machine}.tgz"

    # Prerequisites for XIMEA's own install script (per XIMEA's Ubuntu
    # troubleshooting instructions).
    run_checked(["sudo", "apt-get", "update"])
    run_checked(
        ["sudo", "apt-get", "install", "--yes", "build-essential", f"linux-headers-{platform.uname().release}"]
    )
    try:
        run_checked(["sudo", "apt-get", "install", "--yes", "libtiff5"])
    except subprocess.CalledProcessError:
        # libtiff5 was dropped from newer Ubuntu (e.g. 24.04); shim the old
        # SONAME onto whatever libtiff is actually installed.
        run_checked(["sudo", "apt-get", "install", "--yes", "libtiff6"])
        ldconfig = run_checked(["ldconfig", "-p"], stdout=subprocess.PIPE)
        libtiff6 = next(
            (
                line.rsplit(" => ", 1)[1].strip()
                for line in ldconfig.stdout.splitlines()
                if "libtiff.so.6" in line and " => " in line
            ),
            "",
        )
        if not libtiff6:
            raise RuntimeError("Could not locate libtiff.so.6 with ldconfig")
        run_checked(["sudo", "ln", "-sfn", libtiff6, "/usr/lib/libtiff.so.5"])
        run_checked(["sudo", "ldconfig"])

    download(sdk_url, archive)
    extract_tgz(archive, extract_dir)

    package_root = extract_dir / "package"
    version_file = package_root / "version_LINUX_SP.txt"
    if version_file.is_file():
        print(f"Downloaded XIMEA Linux SDK {version_file.read_text(encoding='utf-8').strip()}")

    # Let XIMEA's own installer handle drivers, libraries and udev rules.
    install_cmd = ["sudo", "./install"]
    if pcie:
        install_cmd.append("-pcie")
    run_checked(install_cmd, cwd=package_root)

    values = {
        "XIMEA_ROOT": "/opt/XIMEA",
        "PYTHONPATH": (
            f"{package_root / 'api' / 'Python' / 'v3'}"
            f"{os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''}"
        ),
    }
    os.environ.update(values)
    append_github_env(values)
    print(f"Installed latest XIMEA SDK beta for Linux {machine} via ./install")


def macos_config() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine == "x86_64":
        return MACOS_X64_URL, "x86_64"
    if machine == "arm64":
        return MACOS_ARM64_URL, "arm64"
    raise RuntimeError(f"Unsupported macOS architecture: {platform.machine()}")


def install_macos() -> None:
    sdk_url, _expected_arch = macos_config()
    runner_temp = Path(require_env("RUNNER_TEMP"))
    download_dir = runner_temp / "ximea-download"
    mount_dir = runner_temp / "ximea-volume"
    machine = platform.machine()
    dmg = download_dir / f"XIMEA_macOS_SP_beta_{machine}.dmg"

    mount_dir.mkdir(parents=True, exist_ok=True)
    download(sdk_url, dmg)

    attached = False
    try:
        run_checked(["hdiutil", "attach", "-nobrowse", "-mountpoint", str(mount_dir), str(dmg)])
        attached = True

        install_script = mount_dir / "install"
        if not install_script.is_file():
            raise FileNotFoundError(f"Missing install script on volume: {install_script}")

        # Let XIMEA's own installer handle the framework, CamTool and
        # system-extension registration.
        run_checked(["sudo", "./install"], cwd=mount_dir)

        # Python bindings live only in the package's example sources, so
        # copy them out before the volume is detached.
        python_src = mount_dir / "Examples" / "xiPython" / "v3" / "ximea"
        python_root = runner_temp / "ximea-python"
        if python_src.is_dir():
            shutil.copytree(python_src, python_root / "ximea", dirs_exist_ok=True)
    finally:
        if attached:
            subprocess.run(
                ["hdiutil", "detach", str(mount_dir)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    values = {
        "XIMEA_ROOT": "/Library/Frameworks/m3api.framework",
        "DYLD_FRAMEWORK_PATH": (
            f"/Library/Frameworks"
            f"{os.pathsep + os.environ['DYLD_FRAMEWORK_PATH'] if os.environ.get('DYLD_FRAMEWORK_PATH') else ''}"
        ),
    }
    python_root = runner_temp / "ximea-python"
    if python_root.is_dir():
        values["PYTHONPATH"] = (
            f"{python_root}{os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''}"
        )

    os.environ.update(values)
    append_github_env(values)
    print(f"Installed latest XIMEA SDK beta for macOS {machine} via ./install")


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
    print(f"Installed latest XIMEA SDK beta for Windows x64 at {sdk_root}")


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
    parser.add_argument("--pcie", action="store_true", help="Pass -pcie to the Linux ./install script.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    selected_platform = args.platform or detect_platform()
    try:
        if selected_platform == "linux":
            install_linux(pcie=args.pcie)
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