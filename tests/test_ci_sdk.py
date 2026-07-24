from __future__ import annotations

import importlib.util
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.support import REPO_ROOT


VERIFY_SCRIPT = REPO_ROOT / "scripts" / "ci" / "verify-ximea-sdk.py"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk.py"


def load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_ximea_sdk", VERIFY_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_install_module():
    spec = importlib.util.spec_from_file_location("install_ximea_sdk", INSTALL_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VerifyXimeaSdkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verify = load_verify_module()

    def touch(self, root: Path, relative: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    def completed(self, stdout: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["fixture"], 0, stdout=stdout, stderr="")

    def test_windows_verification_fails_when_required_assembly_is_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            root = Path(tmp_text)
            for relative in (
                "API/xiAPI/xiApi.h",
                "API/xiAPI/xiapi64.lib",
                "API/xiAPI/xiapi64.dll",
                "Examples/Sources/_libs/xiAPIplus/xiapiplus.h",
                "API/Python/v3/ximea/xiapi.py",
            ):
                self.touch(root, relative)

            with self.assertRaises(FileNotFoundError) as context:
                self.verify.verify_windows(root, sys.executable)

        self.assertIn("xiAPI.NET x64 assembly", str(context.exception))

    def test_windows_verification_imports_expected_python_version(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            root = Path(tmp_text)
            for relative in (
                "API/xiAPI/xiApi.h",
                "API/xiAPI/xiapi64.lib",
                "API/xiAPI/xiapi64.dll",
                "Examples/Sources/_libs/xiAPIplus/xiapiplus.h",
                "API/xiAPI.NET.NET.7.0/xiApi.NETX64.dll",
                "API/Python/v3/ximea/xiapi.py",
            ):
                self.touch(root, relative)
            with mock.patch.object(self.verify, "run_checked", return_value=self.completed("4.99.0\nxiapi.py\n")) as run_checked:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.verify.verify_windows(root, sys.executable)

        run_checked.assert_called_once()

    def test_python_version_must_be_reported(self) -> None:
        with mock.patch.object(self.verify, "run_checked", return_value=self.completed("")):
            with self.assertRaises(RuntimeError) as context:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.verify.verify_python_import(sys.executable)

        self.assertIn("did not report a version", str(context.exception))

    def test_windows_default_sdk_root_prefers_process_environment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            with mock.patch.dict(os.environ, {"XIMEA_SP_PATH": tmp_text}, clear=True), \
                 mock.patch.object(self.verify, "read_windows_environment") as read_environment:
                self.assertEqual(self.verify.default_sdk_root("windows-x64"), Path(tmp_text))

        read_environment.assert_not_called()

    def test_windows_default_sdk_root_reads_installer_environment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(self.verify, "read_windows_environment", return_value=tmp_text):
                self.assertEqual(self.verify.default_sdk_root("windows-x64"), Path(tmp_text))

    def test_windows_default_sdk_root_fails_clearly_when_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.verify, "read_windows_environment", return_value=""):
            with self.assertRaises(RuntimeError) as context:
                self.verify.default_sdk_root("windows-x64")

        self.assertIn("XIMEA_SP_PATH is not set", str(context.exception))

    def test_non_windows_default_sdk_root_fails_clearly_when_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as context:
                self.verify.default_sdk_root("linux-x64")

        self.assertIn("XIMEA_ROOT is not set", str(context.exception))

    def test_linux_verification_checks_architecture_specific_layout_and_cmake(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            root = Path(tmp_text)
            for relative in (
                "include/xiApi.h",
                "include/wintypedefs.h",
                "include/m3Identify.h",
                "include/m3api/wintypedefs.h",
                "include/m3api/m3Identify.h",
                "samples/_libs/xiAPIplus/xiapiplus.h",
                "samples/_libs/xiAPIplus/xiAPIplus_core.cpp",
                "samples/_libs/os_common_header.h",
                "lib/libm3api.so.2",
            ):
                self.touch(root, relative)
            real_require_file = self.verify.require_file

            def require_file(path: Path, description: str) -> None:
                if str(path) == "/usr/lib/libm3api.so.2":
                    return
                real_require_file(path, description)

            with mock.patch.object(self.verify, "require_file", side_effect=require_file), \
                 mock.patch.object(self.verify, "verify_cmake_probe") as cmake_probe, \
                 mock.patch.object(self.verify, "verify_python_import") as python_import:
                self.verify.verify_linux(root, REPO_ROOT, sys.executable, "cmake")

        cmake_probe.assert_called_once()
        python_import.assert_called_once()

    def test_cmake_helper_does_not_create_xiapiplus_static_library_without_cxx(self) -> None:
        text = (REPO_ROOT / "cmake" / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("CMAKE_CXX_COMPILER_LOADED", text)
        self.assertIn("add_library(XIMEA_xiAPIplus STATIC", text)
        self.assertIn("target_compile_definitions(XIMEA_xiAPIplus PRIVATE UNIX)", text)

    def test_macos_verification_rejects_wrong_framework_architecture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            base = Path(tmp_text)
            sdk_root = base / "sdk"
            framework_root = base / "Frameworks"
            for path in (
                framework_root / "m3api.framework" / "Headers" / "xiApi.h",
                framework_root / "m3api.framework" / "m3api",
                sdk_root / "include" / "wintypedefs.h",
                sdk_root / "include" / "m3api" / "wintypedefs.h",
                sdk_root / "Examples" / "Sources" / "_libs" / "xiAPIplus" / "xiapiplus.h",
                sdk_root / "Examples" / "Sources" / "_libs" / "xiAPIplus" / "xiAPIplus_core.cpp",
                sdk_root / "Examples" / "Sources" / "_libs" / "os_common_header.h",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")

            def run_checked(cmd: list[str], **_kwargs):
                if cmd[0] == "lipo":
                    return self.completed("x86_64\n")
                return self.completed("")

            with mock.patch.object(self.verify, "run_checked", side_effect=run_checked), \
                 mock.patch.object(self.verify, "verify_cmake_probe"), \
                 mock.patch.object(self.verify, "verify_python_import"):
                with self.assertRaises(RuntimeError) as context:
                    self.verify.verify_macos(sdk_root, framework_root, REPO_ROOT, sys.executable, "cmake", "arm64")

        self.assertIn("does not contain arm64", str(context.exception))

    def test_macos_signature_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            base = Path(tmp_text)
            sdk_root = base / "sdk"
            framework_root = base / "Frameworks"
            for path in (
                framework_root / "m3api.framework" / "Headers" / "xiApi.h",
                framework_root / "m3api.framework" / "m3api",
                sdk_root / "include" / "wintypedefs.h",
                sdk_root / "include" / "m3api" / "wintypedefs.h",
                sdk_root / "Examples" / "Sources" / "_libs" / "xiAPIplus" / "xiapiplus.h",
                sdk_root / "Examples" / "Sources" / "_libs" / "xiAPIplus" / "xiAPIplus_core.cpp",
                sdk_root / "Examples" / "Sources" / "_libs" / "os_common_header.h",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            failure = subprocess.CalledProcessError(1, ["codesign"])
            with mock.patch.object(self.verify, "run_checked", side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.verify.verify_macos(sdk_root, framework_root, REPO_ROOT, sys.executable, "cmake", "x86_64")

    def test_install_scripts_use_latest_beta_without_checksum_pins(self) -> None:
        text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("getattachment", text)
        self.assertIn("XIMEA_SP_PATH", text)
        self.assertIn("latest", text.lower())
        self.assertNotIn("4.33", text)
        self.assertNotRegex(text.lower(), r"sha256|checksum")
        self.assertIn("XIMEA_Windows_SP_Beta.exe", text)
        self.assertIn("Get-AuthenticodeSignature", text)
        self.assertIn("Import-Module Microsoft.PowerShell.Security", text)
        self.assertNotIn("apt-get", text)
        self.assertNotIn("libtiff", text)
        self.assertIn("./install", text)
        self.assertNotIn("-silent", text)
        self.assertNotIn("-nonet", text)
        self.assertNotIn("copy_headers", text)

    def test_single_python_installer_replaces_shell_and_powershell_wrappers(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "development-ci.yml").read_text(encoding="utf-8")
        self.assertIn("python3 scripts/ci/install-ximea-sdk.py --platform linux", workflow)
        self.assertIn("python scripts/ci/install-ximea-sdk.py --platform windows", workflow)
        self.assertIn("python3 scripts/ci/install-ximea-sdk.py --platform macos", workflow)
        self.assertNotIn("install-ximea-sdk-linux.sh", workflow)
        self.assertNotIn("install-ximea-sdk-macos.sh", workflow)
        self.assertNotIn("install-ximea-sdk-windows.ps1", workflow)
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk-linux.sh").exists())
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk-macos.sh").exists())
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk-windows.ps1").exists())

    def test_installer_detects_architecture_specific_linux_and_macos_urls(self) -> None:
        installer = load_install_module()
        with mock.patch.object(installer.platform, "machine", return_value="x86_64"):
            linux_url = installer.linux_config()
            macos_url, macos_expected_arch = installer.macos_config()
        self.assertIn("ximea_linux_sp_beta.tgz", linux_url)
        self.assertIn("XIMEA_macOX_SP.dmg", macos_url)
        self.assertEqual(macos_expected_arch, "x86_64")

        with mock.patch.object(installer.platform, "machine", return_value="arm64"):
            linux_url = installer.linux_config()
            macos_url, macos_expected_arch = installer.macos_config()
        self.assertIn("ximea_linux_arm_sp_beta.tgz", linux_url)
        self.assertIn("XIMEA_macOS_ARM_SP.dmg", macos_url)
        self.assertEqual(macos_expected_arch, "arm64")

    def test_installer_finds_macos_install_script_when_dmg_wraps_it_in_subdirectory(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            mount_dir = Path(tmp_text)
            package_root = mount_dir / "XIMEA_macOS_SP"
            package_root.mkdir(parents=True)
            install_script = package_root / "install"
            install_script.write_text("#!/bin/sh\n", encoding="utf-8")

            self.assertEqual(installer.find_macos_install_script(mount_dir), install_script)

    def test_installer_uses_macos_app_resource_script_instead_of_gui_binary(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            mount_dir = Path(tmp_text)
            app_dir = mount_dir / "install.app" / "Contents"
            gui_binary = app_dir / "MacOS" / "install"
            resource_script = app_dir / "Resources" / "script"
            gui_binary.parent.mkdir(parents=True)
            resource_script.parent.mkdir(parents=True)
            gui_binary.write_text("gui launcher", encoding="utf-8")
            resource_script.write_text("#!/bin/sh\n", encoding="utf-8")

            self.assertEqual(installer.find_macos_install_script(mount_dir), resource_script)

    def test_installer_rejects_macos_gui_binary_without_resource_script(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            mount_dir = Path(tmp_text)
            gui_binary = mount_dir / "install.app" / "Contents" / "MacOS" / "install"
            gui_binary.parent.mkdir(parents=True)
            gui_binary.write_text("gui launcher", encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as context:
                installer.find_macos_install_script(mount_dir)

        self.assertIn("non-GUI XIMEA macOS install script", str(context.exception))

    def test_installer_prefers_pwsh_for_windows_signature_checks(self) -> None:
        installer = load_install_module()
        with mock.patch.object(installer.shutil, "which", side_effect=lambda name: "C:/Program Files/PowerShell/7/pwsh.exe" if name == "pwsh" else "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"):
            command = installer.power_shell_command("fixture")
        self.assertEqual(command[0], "C:/Program Files/PowerShell/7/pwsh.exe")
        self.assertIn("-NoProfile", command)
        self.assertIn("-NonInteractive", command)
        self.assertEqual(command[-1], "fixture")

    def test_installer_embeds_powershell_values_as_quoted_literals(self) -> None:
        installer = load_install_module()
        self.assertEqual(installer.power_shell_literal("C:/Program Files/XIMEA"), "'C:/Program Files/XIMEA'")
        self.assertEqual(installer.power_shell_literal("C:/Vendor's/XIMEA"), "'C:/Vendor''s/XIMEA'")

    def test_windows_installer_exports_cmake_root_without_reexporting_ximea_sp_path(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            sdk_root = Path(tmp_text) / "XIMEA"
            with mock.patch.dict(os.environ, {"RUNNER_TEMP": tmp_text}, clear=True), \
                 mock.patch.object(installer, "download"), \
                 mock.patch.object(installer, "validate_windows_signature"), \
                 mock.patch.object(installer, "run_checked", return_value=self.completed()), \
                 mock.patch.object(installer, "read_windows_environment", return_value=str(sdk_root)), \
                 mock.patch.object(installer, "append_github_env") as append_env, \
                 mock.patch.object(installer, "append_github_path"):
                with contextlib.redirect_stdout(io.StringIO()):
                    installer.install_windows()

        append_env.assert_called_once_with({
            "XIMEA_ROOT": str(sdk_root),
            "PYTHONPATH": str(sdk_root / "API" / "Python" / "v3"),
        })
        self.assertNotIn("XIMEA_SP_PATH", append_env.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
