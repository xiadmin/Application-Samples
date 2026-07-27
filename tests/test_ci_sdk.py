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


INSTALL_SCRIPT = REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk.py"


def load_install_module():
    spec = importlib.util.spec_from_file_location("install_ximea_sdk", INSTALL_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InstallXimeaSdkTests(unittest.TestCase):
    def completed(self, stdout: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["fixture"], 0, stdout=stdout, stderr="")

    def test_cmake_helper_does_not_create_xiapiplus_static_library_without_cxx(self) -> None:
        text = (REPO_ROOT / "cmake" / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("CMAKE_CXX_COMPILER_LOADED", text)
        self.assertIn("add_library(XIMEA_xiAPIplus STATIC", text)
        self.assertIn("target_compile_definitions(XIMEA_xiAPIplus PRIVATE UNIX)", text)
        self.assertIn("set(_ximea_required_vars XIMEA_LIBRARY XIMEA_INCLUDE_DIR)", text)
        self.assertIn("list(APPEND _ximea_required_vars XIMEA_INCLUDE_PLUS_DIR)", text)
        self.assertIn("if(XIMEA_FOUND AND CMAKE_CXX_COMPILER_LOADED AND NOT TARGET XIMEA::xiAPIplus)", text)

    def test_cmake_helper_uses_ximea_sp_path_as_a_root_hint_on_every_platform(self) -> None:
        text = (REPO_ROOT / "cmake" / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn('set(_ximea_root_hints "${_ximea_sp_path}"', text)

    def test_cmake_helper_uses_macos_xiapiplus_layout_from_ximea_sp_path(self) -> None:
        text = (REPO_ROOT / "cmake" / "CMakeLists.txt").read_text(encoding="utf-8")
        apple_block = text.split("elseif(APPLE)", 1)[1].split("else()", 1)[0]

        self.assertIn(
            'set(_ximea_inc_plus_hints "${_ximea_sp_path}/Examples/Sources/_libs/xiAPIplus"',
            apple_block,
        )

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
        self.assertIn("--sample samples/xiapi/cross-platform/capture-10-images", workflow)
        self.assertNotIn("install-ximea-sdk-linux.sh", workflow)
        self.assertNotIn("install-ximea-sdk-macos.sh", workflow)
        self.assertNotIn("install-ximea-sdk-windows.ps1", workflow)
        self.assertNotIn("verify-ximea-sdk.py", workflow)
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk-linux.sh").exists())
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk-macos.sh").exists())
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "install-ximea-sdk-windows.ps1").exists())
        self.assertFalse((REPO_ROOT / "scripts" / "ci" / "verify-ximea-sdk.py").exists())

    def test_installer_python_smoke_check_imports_xiapi_and_reports_location(self) -> None:
        installer = load_install_module()
        with mock.patch.object(installer, "run_checked", return_value=self.completed()) as run_checked:
            installer.verify_python_import()

        run_checked.assert_called_once_with([
            sys.executable,
            "-c",
            "import ximea; from ximea import xiapi; print(ximea.__version__); print(xiapi.__file__)",
        ])

    def test_each_platform_installer_runs_python_smoke_check(self) -> None:
        text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        for function_name in ("install_linux", "install_macos", "install_windows"):
            function_text = text.split(f"def {function_name}() -> None:", 1)[1].split("\ndef ", 1)[0]
            self.assertIn("verify_python_import()", function_text)

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

    def test_installer_uses_current_macos_app_install_script_instead_of_gui_binary(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            mount_dir = Path(tmp_text)
            app_dir = mount_dir / "install.app" / "Contents"
            gui_binary = app_dir / "MacOS" / "install"
            install_script = app_dir / "MacOS" / "install.sh"
            gui_binary.parent.mkdir(parents=True)
            gui_binary.write_text("gui launcher", encoding="utf-8")
            install_script.write_text("#!/bin/sh\n", encoding="utf-8")

            self.assertEqual(installer.find_macos_install_script(mount_dir), install_script)

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

    def test_macos_installer_runs_read_only_volume_script_through_bash(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            install_script = Path("/Volumes/XIMEA/install.app/Contents/MacOS/install.sh")

            def run_checked(cmd: list[str], **_kwargs):
                if cmd[0] == "lipo":
                    return self.completed("x86_64\n")
                return self.completed()

            with mock.patch.dict(os.environ, {"RUNNER_TEMP": tmp_text}, clear=True), \
                 mock.patch.object(installer, "download"), \
                 mock.patch.object(installer, "find_macos_install_script", return_value=install_script), \
                 mock.patch.object(installer, "install_python_package") as install_python, \
                 mock.patch.object(installer, "append_github_env") as append_env, \
                 mock.patch.object(installer, "run_checked", side_effect=run_checked) as checked, \
                 mock.patch.object(installer.subprocess, "run"):
                with contextlib.redirect_stdout(io.StringIO()):
                    installer.install_macos()

        commands = [call.args[0] for call in checked.call_args_list]
        self.assertIn(["/bin/bash", str(install_script)], commands)
        self.assertFalse(any(command[0] == "chmod" for command in commands))
        attach_command = commands[0]
        self.assertEqual(attach_command[:4], ["hdiutil", "attach", "-readonly", "-nobrowse"])
        self.assertNotIn("-mountpoint", attach_command)
        self.assertIn([
            sys.executable,
            "-c",
            "import ximea; from ximea import xiapi; print(ximea.__version__); print(xiapi.__file__)",
        ], commands)
        install_python.assert_not_called()
        append_env.assert_called_once_with({"XIMEA_SP_PATH": "/Library/Frameworks/m3api.framework"})

    def test_installer_copies_python_package_into_active_interpreter(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            root = Path(tmp_text)
            source = root / "source" / "ximea"
            purelib = root / "site-packages"
            source.mkdir(parents=True)
            (source / "xiapi.py").write_text("fixture\n", encoding="utf-8")

            with mock.patch.object(installer.sysconfig, "get_paths", return_value={"purelib": str(purelib)}):
                installer.install_python_package(source)

            self.assertEqual((purelib / "ximea" / "xiapi.py").read_text(encoding="utf-8"), "fixture\n")

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

    def test_windows_installer_exports_only_ximea_sp_path(self) -> None:
        installer = load_install_module()
        with tempfile.TemporaryDirectory(prefix="ximea-sdk-test-") as tmp_text:
            sdk_root = Path(tmp_text) / "XIMEA"
            with mock.patch.dict(os.environ, {"RUNNER_TEMP": tmp_text}, clear=True), \
                 mock.patch.object(installer, "download"), \
                 mock.patch.object(installer, "validate_windows_signature"), \
                 mock.patch.object(installer, "run_checked", return_value=self.completed()) as checked, \
                 mock.patch.object(installer, "read_windows_environment", return_value=str(sdk_root)), \
                 mock.patch.object(installer, "install_python_package") as install_python, \
                 mock.patch.object(installer, "append_github_env") as append_env:
                with contextlib.redirect_stdout(io.StringIO()):
                    installer.install_windows()

        append_env.assert_called_once_with({
            "XIMEA_SP_PATH": str(sdk_root),
        })
        install_python.assert_called_once_with(sdk_root / "API" / "Python" / "v3" / "ximea")
        commands = [call.args[0] for call in checked.call_args_list]
        self.assertIn([
            sys.executable,
            "-c",
            "import ximea; from ximea import xiapi; print(ximea.__version__); print(xiapi.__file__)",
        ], commands)


if __name__ == "__main__":
    unittest.main()
