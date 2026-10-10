import importlib
import json
import sys
import types
import unittest

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

from evaluation.windows_metadata_integrity import windows_metadata_is_gradable

evaluation = importlib.import_module("evaluation.evaluation")


class WindowsMetadataIntegrityTests(unittest.TestCase):
    def test_detects_observed_captured_go_json_expected_name(self):
        gradable, invalid = windows_metadata_is_gradable(
            {
                "PASS_TO_PASS": ['TestVarzHandler/int_with_type_counter (0.00s)\\n"}'],
                "FAIL_TO_PASS": ["TestStable"],
            },
            "windows",
        )
        self.assertFalse(gradable)
        self.assertEqual(invalid, ['TestVarzHandler/int_with_type_counter (0.00s)\\n"}'])

    def test_detects_observed_package_test_separator_artifact(self):
        malformed = "github.com/docker/docker/libnetwork/: TestSortByNetworkType"
        gradable, invalid = windows_metadata_is_gradable(
            {"PASS_TO_PASS": ["TestStable"], "FAIL_TO_PASS": [malformed]},
            "windows",
        )
        self.assertFalse(gradable)
        self.assertEqual(invalid, [malformed])

    def test_detects_observed_rust_terminal_fragment_but_keeps_harness_suffix(self):
        malformed = "test_advance_bookmarks::test_advance_bookmarks_multiple_bookmarks::commit ."
        valid_harness_name = "test_rewrite::test_rebase_descendants_multiple_swap - should panic"
        gradable, invalid = windows_metadata_is_gradable(
            {"PASS_TO_PASS": [malformed, valid_harness_name], "FAIL_TO_PASS": ["TestRegression"]},
            "windows",
        )
        self.assertFalse(gradable)
        self.assertEqual(invalid, [malformed])

    def test_detects_wrapped_csharp_qualified_prefix_without_rejecting_parameter_newlines(self):
        malformed = (
            "AzureMcp.Tests.Areas.Aks.UnitTest\n"
            "nts.Cluster.ClusterListCommandTests.ExecuteAsync_ValidatesInputCorrectly"
        )
        valid_parameter_newline = "Namespace.Type.Method(value: \"first\nsecond\")"
        gradable, invalid = windows_metadata_is_gradable(
            {
                "PASS_TO_PASS": [malformed, valid_parameter_newline],
                "FAIL_TO_PASS": ["TestRegression"],
            },
            "windows",
        )
        self.assertFalse(gradable)
        self.assertEqual(invalid, [malformed])

    def test_does_not_block_clean_windows_or_linux_metadata(self):
        clean = {"PASS_TO_PASS": ["TestStable"], "FAIL_TO_PASS": ["TestRegression"]}
        self.assertEqual(windows_metadata_is_gradable(clean, "windows"), (True, []))
        self.assertEqual(windows_metadata_is_gradable(clean, "linux"), (True, []))

    def test_evaluation_marks_invalid_windows_metadata_unavailable_before_runtime(self):
        class RuntimeMustNotStart:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                raise AssertionError("runtime must not start for ungradable metadata")

        original_runtime = evaluation.SetupRuntime
        evaluation.SetupRuntime = RuntimeMustNotStart
        try:
            with self.subTest("run_instance"):
                import tempfile
                from pathlib import Path

                with tempfile.TemporaryDirectory() as output_dir:
                    instance = {
                        "instance_id": "example__bad-windows-metadata",
                        "docker_image": "example-image",
                        "test_patch": "test-patch",
                        "pred_patch": "solution-patch",
                        "PASS_TO_PASS": ['TestWrapped (0.00s)\\n"}'],
                        "FAIL_TO_PASS": [],
                    }
                    report = evaluation.run_instance(instance, "windows", output_dir, overwrite=True)
                    self.assertIsNone(report["resolved"])
                    self.assertEqual(report["grading_status"], "unavailable")
                    self.assertEqual(report["failure_class"], "invalid_windows_expected_test_metadata")
                    saved = json.loads(
                        (Path(output_dir) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(saved, report)
        finally:
            evaluation.SetupRuntime = original_runtime


if __name__ == "__main__":
    unittest.main()