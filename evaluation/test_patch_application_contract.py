import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from . import evaluation


class _Result:
    def __init__(self, output="", exit_code=0):
        self.output = output
        self.metadata = types.SimpleNamespace(exit_code=exit_code)


class _Container:
    def __init__(self, patch_results):
        self.patch_results = list(patch_results)
        self.commands = []
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return self.patch_results.pop(0)

    def send_command(self, command):
        self.commands.append(command)
        return _Result()

    def cleanup(self):
        self.cleaned = True


class PatchApplicationContractTests(unittest.TestCase):
    def test_crlf_patch_transport_is_normalized(self):
        patch = "diff --git a/a b/a\r\n@@ -1 +1 @@\r\n-old\r\n+new\r\n"
        normalized, metadata = evaluation.normalize_patch_transport(
            patch, patch_kind="candidate", instance_id="example__crlf"
        )
        self.assertNotIn("\r", normalized)
        self.assertEqual(metadata["line_ending"], "crlf")
        self.assertTrue(metadata["normalized"])

    def test_bare_carriage_return_is_rejected(self):
        with self.assertRaisesRegex(
            evaluation.EvaluationInfrastructureError, "ambiguous_patch_line_endings"
        ):
            evaluation.normalize_patch_transport(
                "diff --git a/a b/a\r\n+bad\rtext\n",
                patch_kind="candidate",
                instance_id="example__ambiguous",
            )

    def setUp(self):
        self.original_runtime = evaluation.SetupRuntime

    def tearDown(self):
        evaluation.SetupRuntime = self.original_runtime

    def _assert_stops_before_commands(self, patch_results, expected_error, expected_phase):
        container = _Container(patch_results)

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        evaluation.SetupRuntime = _Runtime
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(evaluation.EvaluationInfrastructureError, expected_error) as ctx:
                evaluation.evaluate_instance(
                    instance_id="example__required-patch",
                    image="example-image",
                    rebuild_cmd="rebuild",
                    test_cmd="run-tests",
                    print_cmd="print-results",
                    parser="pytest",
                    test_patch="test-patch",
                    solution_patch="solution-patch",
                    platform="windows",
                    output_dir=output_dir,
                )
        self.assertEqual(ctx.exception.failure_class, "required_patch_application_failed")
        self.assertEqual(ctx.exception.failure_phase, expected_phase)
        self.assertEqual(container.commands, [])
        self.assertTrue(container.cleaned)

    def test_failed_test_patch_stops_before_candidate_or_commands(self):
        self._assert_stops_before_commands(
            [False], "test patch failed to apply", "test_patch_from_image_base"
        )

    def test_failed_solution_patch_stops_before_commands(self):
        self._assert_stops_before_commands(
            [True, False], "solution patch failed to apply", "candidate_patch_after_test_patch"
        )

    def test_image_base_mismatch_stops_before_patch_application(self):
        container = _Container([True, True])

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        evaluation.SetupRuntime = _Runtime
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(
                evaluation.EvaluationInfrastructureError, "image_base_commit_mismatch"
            ) as ctx:
                evaluation.evaluate_instance(
                    instance_id="example__base-mismatch",
                    image="example-image",
                    rebuild_cmd="rebuild",
                    test_cmd="run-tests",
                    print_cmd="print-results",
                    parser="pytest",
                    test_patch="test-patch",
                    solution_patch="solution-patch",
                    platform="windows",
                    output_dir=output_dir,
                    base_commit="expected-base",
                )
        self.assertEqual(ctx.exception.failure_class, "image_base_commit_mismatch")
        self.assertEqual(ctx.exception.failure_phase, "image_base_commit_validation")
        self.assertEqual(container.patch_results, [True, True])
        self.assertTrue(container.cleaned)

    def test_run_instance_records_patch_failure_as_not_evaluated(self):
        instance = {
            "instance_id": "example__required-patch",
            "docker_image": "example-image",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test-patch",
            "pred_patch": "solution-patch",
            "log_parser": "pytest",
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": [],
        }
        error = evaluation.EvaluationInfrastructureError(
            "required_patch_application_failed",
            "test patch failed to apply for example__required-patch",
        )
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "evaluate_instance", side_effect=error):
                report = evaluation.run_instance(instance, "linux", output_dir, overwrite=True)
            saved = json.loads(
                (Path(output_dir) / instance["instance_id"] / "report.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "not_evaluated")
        self.assertEqual(report["failure_class"], "required_patch_application_failed")
        self.assertEqual(saved, report)

    def test_infrastructure_results_are_counted_separately(self):
        instance = {
            "instance_id": "example__required-patch",
            "pred_patch": "solution-patch",
        }
        report = {
            "instance_id": instance["instance_id"],
            "resolved": None,
            "grading_status": "not_evaluated",
            "failure_class": "required_patch_application_failed",
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "run_instance", return_value=report):
                results = evaluation.run_instances(
                    [instance], "linux", 1, output_dir, overwrite=True
                )
        self.assertEqual(results["success"], 0)
        self.assertEqual(results["failure"], 0)
        self.assertEqual(results["error"], 0)
        self.assertEqual(results["infrastructure_error"], 1)
        self.assertEqual(
            results["infrastructure_error_ids"], ["example__required-patch"]
        )


if __name__ == "__main__":
    unittest.main()