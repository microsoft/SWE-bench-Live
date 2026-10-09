from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("launch", types.ModuleType("launch"))
sys.modules.setdefault("launch.core", types.ModuleType("launch.core"))
runtime = types.ModuleType("launch.core.runtime")
runtime.SetupRuntime = object
sys.modules.setdefault("launch.core.runtime", runtime)
sys.modules.setdefault("launch.scripts", types.ModuleType("launch.scripts"))
parser = types.ModuleType("launch.scripts.parser")
parser.run_parser = lambda *_args, **_kwargs: {}
sys.modules.setdefault("launch.scripts.parser", parser)
datasets = types.ModuleType("datasets")
datasets.load_dataset = lambda *_args, **_kwargs: []
sys.modules.setdefault("datasets", datasets)

from evaluation.windows_go_cgo_prerequisite_contract import (
    windows_go_cgo_prerequisite_failure,
)

evaluation = importlib.import_module("evaluation.evaluation")


JOINT_CGO_SIGNATURE = """\
github.com/gravitational/teleport/lib/pam: build constraints exclude all Go files in C:\\testbed\\lib\\pam
lib\\backend\\lite\\lite.go:1096:16: undefined: sqlite3.Error
lib\\client\\terminal\\terminal_windows.go:167:14: undefined: tncon.Start
"""


class _Result:
    def __init__(self, output: str = "") -> None:
        self.output = output


class _Container:
    def __init__(self, log: str) -> None:
        self.log = log
        self.cleaned_up = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command: str):
        if command == "print-command":
            return _Result(self.log)
        return _Result("")

    def cleanup(self) -> None:
        self.cleaned_up = True


class WindowsGoCgoPrerequisiteContractTests(unittest.TestCase):
    def test_recognizes_only_the_joint_windows_signature(self) -> None:
        evidence = windows_go_cgo_prerequisite_failure(JOINT_CGO_SIGNATURE, "windows")
        self.assertEqual(
            evidence,
            {
                "build_constraints_excluded": [
                    "build constraints exclude all Go files in C:\\testbed\\lib\\pam"
                ],
                "missing_cgo_symbol_families": ["sqlite3", "tncon"],
            },
        )

    def test_does_not_match_partial_signatures_or_non_windows_results(self) -> None:
        self.assertIsNone(windows_go_cgo_prerequisite_failure(JOINT_CGO_SIGNATURE, "linux"))
        self.assertIsNone(
            windows_go_cgo_prerequisite_failure(
                "build constraints exclude all Go files in C:\\testbed\\lib\\pam\n"
                "undefined: sqlite3.Error\n",
                "windows",
            )
        )
        self.assertIsNone(
            windows_go_cgo_prerequisite_failure(
                "ordinary package compile failure\nundefined: tncon.Start\n",
                "windows",
            )
        )

    def test_evaluation_persists_unavailable_before_expected_test_scoring(self) -> None:
        original_runtime = evaluation.SetupRuntime
        container = _Container(JOINT_CGO_SIGNATURE)

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        evaluation.SetupRuntime = _Runtime
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                instance = {
                    "instance_id": "gravitational__teleport-53067",
                    "docker_image": "example-image",
                    "rebuild_cmds": [],
                    "test_cmds": ["test-command"],
                    "print_cmds": ["print-command"],
                    "test_patch": "official-test-patch",
                    "pred_patch": "solution-patch",
                    "PASS_TO_PASS": ["TestUnrelated"],
                    "FAIL_TO_PASS": ["TestDesktopDiscovery"],
                }
                report = evaluation.run_instance(instance, "windows", output_dir, overwrite=True)
                self.assertIsNone(report["resolved"])
                self.assertEqual(report["grading_status"], "unavailable")
                self.assertEqual(
                    report["failure_class"],
                    "windows_go_cgo_prerequisite_unavailable",
                )
                self.assertEqual(
                    report["cgo_prerequisite_failure"]["missing_cgo_symbol_families"],
                    ["sqlite3", "tncon"],
                )
                instance_dir = Path(output_dir) / instance["instance_id"]
                self.assertEqual(
                    json.loads((instance_dir / "report.json").read_text(encoding="utf-8")),
                    report,
                )
                self.assertEqual(
                    json.loads(
                        (instance_dir / "cgo_prerequisite_failure.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                    report["cgo_prerequisite_failure"],
                )
                self.assertTrue(container.cleaned_up)
        finally:
            evaluation.SetupRuntime = original_runtime


if __name__ == "__main__":
    unittest.main()