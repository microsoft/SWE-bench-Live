import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import evaluation


class FakeRuntime:
    def __init__(self, patch_results):
        self.patch_results = iter(patch_results)
        self.patches = []
        self.commands = []
        self.cleaned_up = False
        self.mnt_host = tempfile.gettempdir()
        self.mnt_container = "/mnt_tmp"

    def apply_patch(self, patch, verbose=False):
        self.patches.append((patch, verbose))
        return next(self.patch_results)

    def send_command(self, command):
        self.commands.append(command)
        return SimpleNamespace(output="")

    def cleanup(self):
        self.cleaned_up = True


class FailClosedEvaluationTests(unittest.TestCase):
    def test_test_patch_failure_stops_before_candidate_or_commands(self):
        runtime = FakeRuntime([False])
        with patch.object(evaluation.SetupRuntime, "from_launch_image", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "Benchmark test patch did not apply cleanly"):
                evaluation.evaluate_instance(
                    "example__project-1",
                    "example-image",
                    "rebuild-command",
                    "run-tests",
                    "collect-log",
                    "test-patch",
                    "candidate-patch",
                    "custom",
                    "linux",
                    tempfile.gettempdir(),
                )
        self.assertEqual(runtime.patches, [("test-patch", True)])
        self.assertEqual(runtime.commands, [])
        self.assertTrue(runtime.cleaned_up)

    def test_candidate_patch_failure_stops_before_rebuild_or_tests(self):
        runtime = FakeRuntime([True, False])
        with patch.object(evaluation.SetupRuntime, "from_launch_image", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "Candidate patch did not apply cleanly"):
                evaluation.evaluate_instance(
                    "example__project-1",
                    "example-image",
                    "rebuild-command",
                    "run-tests",
                    "collect-log",
                    "test-patch",
                    "candidate-patch",
                    "custom",
                    "linux",
                    tempfile.gettempdir(),
                )
        self.assertEqual(
            runtime.patches,
            [("test-patch", True), ("candidate-patch", True)],
        )
        self.assertEqual(runtime.commands, [])
        self.assertTrue(runtime.cleaned_up)

    def test_root_change_after_rebuild_stops_before_test(self):
        runtime = FakeRuntime([True, True])
        roots = iter(["/testbed\n", "/different-worktree\n"])

        def send_command(command):
            runtime.commands.append(command)
            if command == "git rev-parse --show-toplevel":
                return SimpleNamespace(output=next(roots), metadata=SimpleNamespace(exit_code=0))
            return SimpleNamespace(output="", metadata=SimpleNamespace(exit_code=0))

        runtime.send_command = send_command
        with patch.object(evaluation.SetupRuntime, "from_launch_image", return_value=runtime):
            with self.assertRaisesRegex(evaluation.EvaluationInfrastructureError, "not patched root"):
                evaluation.evaluate_instance(
                    "example__root-change",
                    "example-image",
                    "rebuild-command",
                    "run-tests",
                    "collect-log",
                    "test-patch",
                    "candidate-patch",
                    "custom",
                    "linux",
                    tempfile.gettempdir(),
                )
        self.assertEqual(
            runtime.commands,
            ["git rev-parse --show-toplevel", "rebuild-command", "git rev-parse --show-toplevel"],
        )
        self.assertTrue(runtime.cleaned_up)

    def test_unmatched_parser_output_is_an_evaluator_contract_error(self):
        instance = {
            "instance_id": "example__project-1",
            "docker_image": "example-image",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test-patch",
            "pred_patch": "candidate-patch",
            "log_parser": "custom",
            "PASS_TO_PASS": ["expected::stable"],
            "FAIL_TO_PASS": ["expected::fixed"],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "validate_solution_patch_format", return_value=None), patch.object(
                evaluation, "evaluate_instance", return_value={"unrelated::test": "pass"}
            ):
                with self.assertRaisesRegex(evaluation.EvaluationContractError, "matched none"):
                    evaluation.run_instance(instance, "linux", output_dir, True)

            report = __import__("json").loads(
                (Path(output_dir) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
            )
        self.assertFalse(report["resolved"])
        self.assertFalse(report["evaluation_contract"]["valid"])
        self.assertEqual(report["evaluation_contract"]["matched_expected_tests"], 0)

    def test_malformed_candidate_patch_is_not_evaluated(self):
        instance = {
            "instance_id": "example__malformed-candidate",
            "docker_image": "example-image",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test-patch",
            "pred_patch": "diff --git a/a b/a\n--- a/a\n+++ b/a",
            "log_parser": "custom",
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": [],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(
                evaluation,
                "evaluate_instance",
                side_effect=AssertionError("malformed candidate must not start evaluation"),
            ):
                report = evaluation.run_instance(instance, "linux", output_dir, True)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "not_evaluated")
        self.assertEqual(report["failure_class"], "local_candidate_patch_preflight_failed")
        self.assertIn("missing_terminal_newline", report["candidate_patch_preflight"]["reasons"])

    def test_failed_test_command_cannot_score_partial_passes(self):
        instance = {
            "instance_id": "example__failed-command",
            "docker_image": "example-image",
            "rebuild_cmds": [],
            "test_cmds": ["run-tests"],
            "print_cmds": ["collect-log"],
            "test_patch": "test-patch",
            "pred_patch": "candidate-patch",
            "log_parser": "custom",
            "PASS_TO_PASS": ["expected::stable"],
            "FAIL_TO_PASS": ["expected::fixed"],
        }
        runtime = FakeRuntime([True, True])

        def send_command(command):
            runtime.commands.append(command)
            if command == "run-tests":
                return SimpleNamespace(output="partial test output", metadata=SimpleNamespace(exit_code=1))
            return SimpleNamespace(output="partial test output", metadata=SimpleNamespace(exit_code=0))

        runtime.send_command = send_command
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation.SetupRuntime, "from_launch_image", return_value=runtime), patch.object(
                evaluation, "validate_solution_patch_format", return_value=None
            ), patch.object(
                evaluation,
                "run_parser",
                return_value={"expected::fixed": "pass", "expected::stable": "skip"},
            ):
                report = evaluation.run_instance(instance, "windows", output_dir, True)
        self.assertFalse(report["resolved"])
        self.assertEqual(report["FAIL_TO_PASS"]["success"], [])
        self.assertEqual(report["FAIL_TO_PASS"]["failure"], ["expected::fixed"])
        self.assertTrue(runtime.cleaned_up)

    def test_failed_go_command_preserves_explicit_terminal_passes(self):
        status = evaluation._demote_unproven_passes_after_failed_command(
            {
                "example.com/project/pkg/TestTarget": "pass",
                "example.com/project/pkg.TestTarget": "pass",
                "example.com/project/pkg::TestTarget": "pass",
                "example.com/project/pkg/TestOther": "pass",
                "example.com/project/pkg/TestSkipped": "skip",
            },
            "\n".join(
                [
                    json.dumps(
                        {
                            "Action": "pass",
                            "Package": "example.com/project/pkg",
                            "Test": "TestTarget",
                            "Elapsed": 0,
                        }
                    ),
                    json.dumps(
                        {
                            "Action": "output",
                            "Package": "example.com/project/pkg",
                            "Output": "FAIL\texample.com/project/pkg\t0.01s\n",
                        }
                    ),
                ]
            ),
        )

        self.assertEqual(
            status,
            {
                "example.com/project/pkg/TestTarget": "pass",
                "example.com/project/pkg.TestTarget": "pass",
                "example.com/project/pkg::TestTarget": "pass",
                "example.com/project/pkg/TestOther": "fail",
                "example.com/project/pkg/TestSkipped": "skip",
            },
        )

    def test_failed_command_preserves_junit_terminal_passes(self):
        log = """<?xml version=\"1.0\"?>
<testsuites><testsuite name=\"pkg\" tests=\"2\" failures=\"1\">
  <testcase classname=\"example/pkg\" name=\"TestKept\" />
  <testcase classname=\"example/pkg\" name=\"TestBroken\"><failure message=\"boom\">boom</failure></testcase>
</testsuite></testsuites>
<?xml version=\"1.0\"?>
<testsuites><testsuite name=\"other\" tests=\"1\" failures=\"0\">
  <testcase classname=\"example/other\" name=\"TestOther\" />
</testsuite></testsuites>"""
        status = evaluation._demote_unproven_passes_after_failed_command(
            {
                "example/pkg::TestKept": "pass",
                "example/pkg::TestBroken": "pass",
                "example/other::TestOther": "pass",
                "unrelated::TestUnknown": "pass",
            },
            log,
        )
        self.assertEqual(
            status,
            {
                "example/pkg::TestKept": "pass",
                "example/pkg::TestBroken": "fail",
                "example/other::TestOther": "pass",
                "unrelated::TestUnknown": "fail",
            },
        )

    def test_matching_parser_output_can_be_scored(self):
        instance = {
            "instance_id": "example__project-1",
            "docker_image": "example-image",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test-patch",
            "pred_patch": "candidate-patch",
            "log_parser": "custom",
            "PASS_TO_PASS": ["expected::stable"],
            "FAIL_TO_PASS": ["expected::fixed"],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "validate_solution_patch_format", return_value=None), patch.object(
                evaluation,
                "evaluate_instance",
                return_value={"expected::stable": "pass", "expected::fixed": "pass"},
            ):
                report = evaluation.run_instance(instance, "linux", output_dir, True)
        self.assertTrue(report["resolved"])
        self.assertTrue(report["evaluation_contract"]["valid"])
        self.assertEqual(report["evaluation_contract"]["matched_expected_tests"], 2)


if __name__ == "__main__":
    unittest.main()