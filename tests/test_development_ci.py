from __future__ import annotations

import re
import unittest
from pathlib import Path

from tests.support import REPO_ROOT


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "development-ci.yml"


class DevelopmentCiWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_triggers_only_on_push_to_development(self) -> None:
        self.assertIn("on:\n  push:\n    branches: [development]", self.text)
        self.assertNotRegex(self.text, r"pull_request|schedule|workflow_dispatch")
        self.assertNotIn("branches: [main]", self.text)

    def test_fixed_github_hosted_runner_labels_cover_all_five_platforms(self) -> None:
        for label in ("ubuntu-24.04", "ubuntu-24.04-arm", "windows-2022", "macos-15-intel", "macos-15"):
            self.assertIn(label, self.text)
        for platform in ("linux-x64", "linux-arm64", "windows-x64", "macos-x64", "macos-arm64"):
            self.assertIn(platform, self.text)
        self.assertNotIn("self-hosted", self.text)

    def test_every_job_installs_and_verifies_sdk_before_building(self) -> None:
        for job_name in ("linux:", "windows:", "macos:"):
            job_start = self.text.index(f"  {job_name}")
            next_job = self.text.find("\n  ", job_start + 3)
            while next_job != -1 and self.text[next_job + 3:next_job + 4] == " ":
                next_job = self.text.find("\n  ", next_job + 1)
            job_text = self.text[job_start: next_job if next_job != -1 else len(self.text)]
            install = job_text.index("Install XIMEA SDK")
            verify = job_text.index("Verify XIMEA SDK")
            build = min(index for index in (
                job_text.find("Build and attempt"),
                job_text.find("Compile-check and attempt"),
            ) if index != -1)
            self.assertLess(install, verify)
            self.assertLess(verify, build)

    def test_build_steps_use_run_with_positive_timeout(self) -> None:
        lines = self.text.splitlines()
        build_commands = [
            " ".join(lines[index:index + 3])
            for index, line in enumerate(lines)
            if "scripts/build.py" in line
        ]
        self.assertGreaterEqual(len(build_commands), 7)
        for command in build_commands:
            self.assertIn("--run", command)
            match = re.search(r"--run-timeout-seconds\s+(\d+)", command)
            self.assertIsNotNone(match, command)
            assert match is not None
            self.assertGreater(int(match.group(1)), 0)

    def test_expected_sample_types_are_selected_by_platform(self) -> None:
        self.assertGreaterEqual(self.text.count("--type cmake --skip-platform-specific"), 3)
        self.assertGreaterEqual(self.text.count("--type python --skip-platform-specific"), 3)
        self.assertIn("--type dotnet", self.text)
        dotnet_line = next(line for line in self.text.splitlines() if "--type dotnet" in line)
        self.assertNotIn("--skip-platform-specific", dotnet_line)

    def test_no_artifact_release_deployment_or_cache_logic_exists(self) -> None:
        forbidden = ("upload-artifact", "download-artifact", "release", "deploy", "cache", "pages")
        lowered = self.text.lower()
        for word in forbidden:
            self.assertNotIn(word, lowered)

    def test_checkout_and_setup_python_actions_are_pinned_to_full_shas(self) -> None:
        uses_lines = [line.strip() for line in self.text.splitlines() if line.strip().startswith("uses:")]
        self.assertTrue(uses_lines)
        for line in uses_lines:
            self.assertRegex(line, r"@[0-9a-f]{40}(?:\s+#\s+v[0-9])?")

    def test_actions_use_node_24_versions(self) -> None:
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", self.text)
        self.assertIn("actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0", self.text)
        self.assertNotIn("actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683", self.text)
        self.assertNotIn("actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", self.text)

    def test_permissions_and_concurrency_are_restrictive(self) -> None:
        self.assertIn("permissions:\n  contents: read", self.text)
        self.assertIn("cancel-in-progress: true", self.text)


if __name__ == "__main__":
    unittest.main()
