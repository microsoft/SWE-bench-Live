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

from evaluation.windows_go_execution_contract import (
    observed_go_test_roots,
    touched_go_test_functions,
    unexecuted_windows_go_test_patch_targets,
)

evaluation = importlib.import_module("evaluation.evaluation")


class _Result:
    def __init__(self, output: str = "") -> None:
        self.output = output


class _Container:
    def __init__(self, log: str) -> None:
        self.log = log

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command: str):
        if command == "print-command":
            return _Result(self.log)
        return _Result("")

    def cleanup(self) -> None:
        return None


class WindowsGoExecutionContractTests(unittest.TestCase):
    PATCH = """diff --git a/x-pack/packetbeat/tests/system/app_test.go b/x-pack/packetbeat/tests/system/app_test.go
index 1111111..2222222 100644
--- a/x-pack/packetbeat/tests/system/app_test.go
+++ b/x-pack/packetbeat/tests/system/app_test.go
@@ -54,8 +54,10 @@ func TestDevices(t *testing.T) {
-\trequire.Contains(t, stdout, ifc.Name)
+\tif !strings.Contains(stdout, ifc.Name) {
+\t\trequire.Contains(t, stdout, address.String())
+\t}
 }
"""

    def test_extracts_a_touched_existing_test_from_hunk_context(self) -> None:
        self.assertEqual(touched_go_test_functions(self.PATCH), ["TestDevices"])

    def test_extracts_a_removed_test_declaration(self) -> None:
        patch = """diff --git a/pkg/example_test.go b/pkg/example_test.go
--- a/pkg/example_test.go
+++ b/pkg/example_test.go
@@ -1,3 +0,0 @@
-func TestRemoved(t *testing.T) {}
"""
        self.assertEqual(touched_go_test_functions(patch), ["TestRemoved"])

    def test_reads_terminal_go_test_roots(self) -> None:
        log = "\n".join([
            '{"Action":"pass","Package":"example/pkg","Test":"TestOther/child"}',
            '{"Action":"fail","Package":"example/pkg","Test":"TestElse"}',
        ])
        self.assertEqual(observed_go_test_roots(log), {"TestOther", "TestElse"})

    def test_rejects_windows_stream_that_executed_none_of_the_official_targets(self) -> None:
        log = '{"Action":"pass","Package":"example/pkg","Test":"TestOther"}\n'
        self.assertEqual(
            unexecuted_windows_go_test_patch_targets(self.PATCH, log, "windows"),
            (["TestDevices"], ["TestDevices"], {"TestOther"}),
        )

    def test_accepts_a_target_or_non_go_stream(self) -> None:
        targeted = '{"Action":"pass","Package":"example/pkg","Test":"TestDevices/address-fallback"}\n'
        self.assertEqual(unexecuted_windows_go_test_patch_targets(self.PATCH, targeted, "windows")[0], [])
        self.assertEqual(unexecuted_windows_go_test_patch_targets(self.PATCH, "ordinary output", "windows")[0], [])
        self.assertEqual(unexecuted_windows_go_test_patch_targets(self.PATCH, "ordinary output", "linux")[0], [])

    def test_evaluation_persists_unavailable_before_expected_test_scoring(self) -> None:
        original_runtime = evaluation.SetupRuntime

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return _Container('{"Action":"pass","Package":"example/pkg","Test":"TestOther"}\n')

        evaluation.SetupRuntime = _Runtime
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                instance = {
                    "instance_id": "example__unexecuted-go-regression",
                    "docker_image": "example-image",
                    "rebuild_cmds": [],
                    "test_cmds": ["test-command"],
                    "print_cmds": ["print-command"],
                    "test_patch": self.PATCH,
                    "pred_patch": "solution-patch",
                    "PASS_TO_PASS": [],
                    "FAIL_TO_PASS": [],
                }
                report = evaluation.run_instance(instance, "windows", output_dir, overwrite=True)
                self.assertIsNone(report["resolved"])
                self.assertEqual(report["grading_status"], "unavailable")
                self.assertEqual(report["failure_class"], "windows_go_test_patch_regression_not_executed")
                saved = json.loads(
                    (Path(output_dir) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved, report)
        finally:
            evaluation.SetupRuntime = original_runtime


if __name__ == "__main__":
    unittest.main()