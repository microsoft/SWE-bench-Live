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

from evaluation.windows_test_patch_contract import (
    added_go_test_functions,
    touched_go_test_functions,
    windows_go_test_patch_contract_is_gradable,
)

evaluation = importlib.import_module("evaluation.evaluation")


class WindowsGoTestPatchContractTests(unittest.TestCase):
    PATCH = """diff --git a/pkg/example_test.go b/pkg/example_test.go
index 1111111..2222222 100644
--- a/pkg/example_test.go
+++ b/pkg/example_test.go
@@ -1,3 +1,8 @@
 package example
+
+func TestRegression(t *testing.T) {}
+
+func TestOtherRegression(t *testing.T) {}
+diff --git a/pkg/example.go b/pkg/example.go
index 1111111..2222222 100644
--- a/pkg/example.go
+++ b/pkg/example.go
@@ -1,2 +1,2 @@
-package example
+package example
"""

    def test_extracts_only_added_top_level_tests_from_go_test_files(self):
        self.assertEqual(
            added_go_test_functions(self.PATCH),
            ["TestRegression", "TestOtherRegression"],
        )

    def test_tracks_existing_test_body_from_hunk_context(self):
        patch = """diff --git a/provisioner/powershell/provisioner_test.go b/provisioner/powershell/provisioner_test.go
--- a/provisioner/powershell/provisioner_test.go
+++ b/provisioner/powershell/provisioner_test.go
@@ -71,7 +71,6 @@ func TestProvisionerPrepare_extractScript(t *testing.T) {
-\tassert.Equal(t, oldScript, actual)
+\tassert.Equal(t, newScript, actual)
 }
"""
        self.assertEqual(touched_go_test_functions(patch), ["TestProvisionerPrepare_extractScript"])

    def test_rejects_existing_test_absent_from_all_published_expected_names(self):
        patch = """diff --git a/provisioner/powershell/provisioner_test.go b/provisioner/powershell/provisioner_test.go
--- a/provisioner/powershell/provisioner_test.go
+++ b/provisioner/powershell/provisioner_test.go
@@ -71,7 +71,6 @@ func TestProvisionerPrepare_extractScript(t *testing.T) {
-\tassert.Equal(t, oldScript, actual)
+\tassert.Equal(t, newScript, actual)
 }
"""
        instance = {
            "test_patch": patch,
            "PASS_TO_PASS": ["github.com/hashicorp/packer/command/TestUnrelated"],
            "FAIL_TO_PASS": ["github.com/hashicorp/packer/fix/TestOther"],
        }
        self.assertEqual(
            windows_go_test_patch_contract_is_gradable(instance, "windows"),
            (False, ["TestProvisionerPrepare_extractScript"]),
        )

    def test_accepts_packer_style_expected_identity_from_pass_to_pass(self):
        patch = """diff --git a/pkg/example_test.go b/pkg/example_test.go
--- a/pkg/example_test.go
+++ b/pkg/example_test.go
@@ -1,3 +1,3 @@ func TestTarget(t *testing.T) {
-\tassert.Equal(t, old, got)
+\tassert.Equal(t, want, got)
 }
"""
        instance = {
            "test_patch": patch,
            "PASS_TO_PASS": ["github.com/example/repo/pkg/TestTarget"],
            "FAIL_TO_PASS": [],
        }
        self.assertEqual(windows_go_test_patch_contract_is_gradable(instance, "windows"), (True, []))

    def test_accepts_matching_test_or_subtest_and_ignores_other_platforms(self):
        instance = {
            "test_patch": self.PATCH,
            "FAIL_TO_PASS": ["example/pkg::TestRegression/edge"],
        }
        self.assertEqual(windows_go_test_patch_contract_is_gradable(instance, "windows"), (True, []))
        self.assertEqual(
            windows_go_test_patch_contract_is_gradable(
                {"test_patch": self.PATCH, "FAIL_TO_PASS": []}, "linux"
            ),
            (True, []),
        )

    def test_rejects_unrepresented_windows_go_test_patch_roots(self):
        instance = {"test_patch": self.PATCH, "FAIL_TO_PASS": ["TestUnrelated"]}
        self.assertEqual(
            windows_go_test_patch_contract_is_gradable(instance, "windows"),
            (False, ["TestRegression", "TestOtherRegression"]),
        )

    def test_evaluation_is_unavailable_before_runtime_when_contract_is_missing(self):
        class RuntimeMustNotStart:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                raise AssertionError("runtime must not start for an unrepresented test patch")

        original_runtime = evaluation.SetupRuntime
        evaluation.SetupRuntime = RuntimeMustNotStart
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                instance = {
                    "instance_id": "example__unrepresented-go-test-patch",
                    "docker_image": "example-image",
                    "test_patch": self.PATCH,
                    "pred_patch": "solution-patch",
                    "PASS_TO_PASS": [],
                    "FAIL_TO_PASS": ["TestUnrelated"],
                }
                report = evaluation.run_instance(instance, "windows", output_dir, overwrite=True)
                self.assertIsNone(report["resolved"])
                self.assertEqual(report["grading_status"], "unavailable")
                self.assertEqual(
                    report["failure_class"],
                    "unrepresented_windows_go_test_patch_regressions",
                )
                saved = json.loads(
                    (Path(output_dir) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved, report)
        finally:
            evaluation.SetupRuntime = original_runtime


if __name__ == "__main__":
    unittest.main()