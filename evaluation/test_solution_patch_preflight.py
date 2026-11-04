import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if "datasets" not in sys.modules:
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: None
    sys.modules["datasets"] = datasets

if "launch.core.runtime" not in sys.modules:
    launch = types.ModuleType("launch")
    launch.__path__ = []
    launch_core = types.ModuleType("launch.core")
    launch_core.__path__ = []
    launch_runtime = types.ModuleType("launch.core.runtime")
    launch_runtime.SetupRuntime = object
    launch_scripts = types.ModuleType("launch.scripts")
    launch_scripts.__path__ = []
    launch_parser = types.ModuleType("launch.scripts.parser")
    launch_parser.run_parser = lambda *_args, **_kwargs: {}
    sys.modules.update({
        "launch": launch,
        "launch.core": launch_core,
        "launch.core.runtime": launch_runtime,
        "launch.scripts": launch_scripts,
        "launch.scripts.parser": launch_parser,
    })

from evaluation import evaluation as evaluator


VALID_PATCH = "\n".join([
    "diff --git a/file.txt b/file.txt",
    "index 1111111..2222222 100644",
    "--- a/file.txt",
    "+++ b/file.txt",
    "@@ -1 +1 @@",
    "-old",
    "+new",
    "",
])
MALFORMED_PATCH = VALID_PATCH.rstrip("\n")


class SolutionPatchPreflightTests(unittest.TestCase):
    def test_accepts_canonical_file_backed_git_patch(self):
        self.assertIsNone(evaluator.validate_solution_patch_format(VALID_PATCH))

    def test_rejects_missing_terminal_newline_and_git_rejection(self):
        evidence = evaluator.validate_solution_patch_format(MALFORMED_PATCH)
        self.assertIsNotNone(evidence)
        self.assertIn("missing_terminal_newline", evidence["reasons"])
        self.assertIn("git_apply_stat_rejected_patch", evidence["reasons"])
        self.assertFalse(evidence["terminal_newline"])

    def test_run_instance_writes_not_evaluated_report_without_starting_container(self):
        instance = {
            "instance_id": "fixture__invalid_candidate_patch",
            "docker_image": "fixture:image",
            "rebuild_cmds": [],
            "test_cmds": ["test-command"],
            "print_cmds": ["print-command"],
            "test_patch": "official test patch",
            "pred_patch": MALFORMED_PATCH,
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": [],
        }

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                raise AssertionError("container must not start for a malformed local candidate artifact")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluator, "SetupRuntime", _Runtime):
                report = evaluator.run_instance(instance, "windows", directory, True)
            saved = json.loads(
                (Path(directory) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
            )
        self.assertEqual(report, saved)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "not_evaluated")
        self.assertEqual(report["failure_class"], "local_candidate_patch_preflight_failed")


if __name__ == "__main__":
    unittest.main()