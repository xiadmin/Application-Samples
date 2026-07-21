from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tests.support import load_build_module, prepend_path, write_executable


class RuntimeAttemptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.build = load_build_module()
        self.sample = self.build.Sample(
            name="fake-sample",
            sample_type=self.build.SampleType.PYTHON,
            path=Path("/tmp/fake-sample"),
            entry=Path("/tmp/fake-sample/main.py"),
            is_platform_specific=False,
        )

    def command(self, argv: list[str]):
        return self.build.RuntimeCommand(self.sample, "fake runnable", argv, Path.cwd())

    def test_passing_process_is_reported_as_passed(self) -> None:
        result = self.build.execute_runtime_command(
            self.command([sys.executable, "-c", "print('hello from runtime')"]),
            timeout_seconds=5,
        )

        self.assertEqual(self.build.RuntimeStatus.PASSED, result.status)
        self.assertEqual(0, result.exit_code)
        self.assertIn("hello from runtime", result.stdout)

    def test_nonzero_process_preserves_captured_output(self) -> None:
        result = self.build.execute_runtime_command(
            self.command([
                sys.executable,
                "-c",
                "import sys; print('out text'); print('err text', file=sys.stderr); sys.exit(7)",
            ]),
            timeout_seconds=5,
        )

        self.assertEqual(self.build.RuntimeStatus.NO_CAMERA_OR_FAILED, result.status)
        self.assertEqual(7, result.exit_code)
        self.assertIn("out text", result.stdout)
        self.assertIn("err text", result.stderr)

    def test_timeout_terminates_child_process(self) -> None:
        result = self.build.execute_runtime_command(
            self.command([sys.executable, "-c", "import time; time.sleep(30)"]),
            timeout_seconds=1,
        )

        self.assertEqual(self.build.RuntimeStatus.TIMED_OUT, result.status)
        self.assertIsNone(result.exit_code)

    def test_missing_command_is_reported_without_raising(self) -> None:
        result = self.build.execute_runtime_command(
            self.command(["definitely-not-a-real-ximea-ci-command"]),
            timeout_seconds=5,
        )

        self.assertEqual(self.build.RuntimeStatus.COULD_NOT_START, result.status)
        self.assertIn("definitely-not-a-real-ximea-ci-command", result.message)

    def test_multiple_commands_continue_after_failure(self) -> None:
        commands = [
            self.command([sys.executable, "-c", "import sys; sys.exit(3)"]),
            self.command([sys.executable, "-c", "print('still ran')"]),
        ]

        results = self.build.run_runtime_commands(commands, timeout_seconds=5)

        self.assertEqual([
            self.build.RuntimeStatus.NO_CAMERA_OR_FAILED,
            self.build.RuntimeStatus.PASSED,
        ], [result.status for result in results])
        self.assertIn("still ran", results[1].stdout)

    def test_github_warning_annotations_are_printed_for_runtime_failures(self) -> None:
        result = self.build.execute_runtime_command(
            self.command([sys.executable, "-c", "import sys; sys.exit(2)"]),
            timeout_seconds=5,
        )

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.build.print_runtime_summary([result])

        self.assertIn("Runtime attempt summary", stream.getvalue())
        self.assertIn("::warning::", stream.getvalue())
        self.assertIn("no_camera_or_failed", stream.getvalue())


class BuildRunCliTests(unittest.TestCase):
    def make_fake_repo(self, tmp: Path) -> Path:
        samples = tmp / "samples"
        cmake_sample = samples / "xiapi" / "cross-platform" / "capture-10-images"
        cmake_sample.mkdir(parents=True)
        (cmake_sample / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n", encoding="utf-8")
        (cmake_sample / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
        return samples

    def make_fake_cmake(self, bin_dir: Path, exit_code: int = 7) -> None:
        write_executable(bin_dir / "cmake", f"""
            #!/usr/bin/env python3
            import pathlib
            import sys
            args = sys.argv[1:]
            if '--build' in args:
                build_dir = pathlib.Path(args[args.index('--build') + 1])
                exe = build_dir / 'build' / 'fake-camera'
                exe.parent.mkdir(parents=True, exist_ok=True)
                exe.write_text('#!/usr/bin/env python3\\nimport sys; print(\"camera attempted\"); sys.exit({exit_code})\\n', encoding='utf-8')
                exe.chmod(0o755)
            elif '-B' in args:
                pathlib.Path(args[args.index('-B') + 1]).mkdir(parents=True, exist_ok=True)
            sys.exit(0)
        """)

    def run_main(self, argv: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        build = load_build_module()
        with mock.patch.object(sys, "argv", ["build.py", *argv]), \
             mock.patch.dict(os.environ, env or os.environ.copy(), clear=True), \
             contextlib.redirect_stdout(stdout), \
             contextlib.redirect_stderr(stderr):
            code = build.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_existing_behavior_without_run_does_not_print_runtime_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-test-") as tmp_text:
            tmp = Path(tmp_text)
            samples = self.make_fake_repo(tmp)
            bin_dir = tmp / "bin"
            self.make_fake_cmake(bin_dir)
            code, stdout, stderr = self.run_main([
                "--all", "--type", "cmake", "--samples-root", str(samples), "--output-root", str(tmp / "out")
            ], prepend_path(bin_dir))

        self.assertEqual(0, code, stderr)
        self.assertIn("Build summary", stdout)
        self.assertNotIn("Runtime attempt summary", stdout)

    def test_runtime_only_failure_does_not_turn_successful_build_into_failed_cli(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-test-") as tmp_text:
            tmp = Path(tmp_text)
            samples = self.make_fake_repo(tmp)
            bin_dir = tmp / "bin"
            self.make_fake_cmake(bin_dir, exit_code=7)
            code, stdout, stderr = self.run_main([
                "--all", "--type", "cmake", "--run", "--run-timeout-seconds", "5",
                "--samples-root", str(samples), "--output-root", str(tmp / "out")
            ], prepend_path(bin_dir))

        self.assertEqual(0, code, stderr)
        self.assertIn("Build summary", stdout)
        self.assertIn("Runtime attempt summary", stdout)
        self.assertIn("no_camera_or_failed", stdout)
        self.assertIn("::warning::", stdout)

    def test_python_compile_failure_prevents_python_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-test-") as tmp_text:
            tmp = Path(tmp_text)
            samples = tmp / "samples"
            sample = samples / "xiapi-python" / "cross-platform" / "bad-python"
            sample.mkdir(parents=True)
            (sample / "main.py").write_text("if True print('bad')\n", encoding="utf-8")
            code, stdout, stderr = self.run_main([
                "--all", "--type", "python", "--run", "--samples-root", str(samples), "--output-root", str(tmp / "out")
            ])

        self.assertNotEqual(0, code)
        self.assertIn("Python syntax check failed", stdout)
        self.assertIn("Build summary", stdout)
        self.assertNotIn("Runtime attempt summary", stdout)
        self.assertIn("SyntaxError", stderr)

    def test_skipped_platform_specific_sample_is_not_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-test-") as tmp_text:
            tmp = Path(tmp_text)
            samples = tmp / "samples"
            sample = samples / "xiapi-python" / "hardware-specific" / "jetson"
            sample.mkdir(parents=True)
            (sample / "main.py").write_text("print('must not run')\n", encoding="utf-8")
            code, stdout, stderr = self.run_main([
                "--all", "--type", "python", "--skip-platform-specific", "--run",
                "--samples-root", str(samples), "--output-root", str(tmp / "out")
            ])

        self.assertEqual(0, code, stderr)
        self.assertIn("platform-specific sample skipped", stdout)
        self.assertNotIn("must not run", stdout)

    def test_zero_and_negative_timeout_values_are_rejected(self) -> None:
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory(prefix="build-test-") as tmp_text:
                    tmp = Path(tmp_text)
                    samples = self.make_fake_repo(tmp)
                    code, _stdout, stderr = self.run_main([
                        "--all", "--run", "--run-timeout-seconds", value,
                        "--samples-root", str(samples), "--output-root", str(tmp / "out")
                    ])
                self.assertNotEqual(0, code)
                self.assertIn("--run-timeout-seconds must be positive", stderr)

    def test_build_failures_still_return_nonzero_with_run_enabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build-test-") as tmp_text:
            tmp = Path(tmp_text)
            samples = self.make_fake_repo(tmp)
            code, stdout, stderr = self.run_main([
                "--all", "--type", "cmake", "--run", "--samples-root", str(samples), "--output-root", str(tmp / "out")
            ], env={"PATH": ""})

        self.assertNotEqual(0, code)
        self.assertIn("cmake CLI not found", stdout)
        self.assertNotIn("Runtime attempt summary", stdout)


if __name__ == "__main__":
    unittest.main()
