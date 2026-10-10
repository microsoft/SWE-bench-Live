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

from evaluation.windows_java_disabled_test_contract import (
    reenabled_windows_junit_test_suffixes,
    windows_java_disabled_test_contract_is_gradable,
)

evaluation = importlib.import_module("evaluation.evaluation")


class WindowsJavaDisabledTestContractTests(unittest.TestCase):
    PATCH = """diff --git a/module/src/test/java/io/example/TracerTest.java b/module/src/test/java/io/example/TracerTest.java
index 1111111..2222222 100644
--- a/module/src/test/java/io/example/TracerTest.java
+++ b/module/src/test/java/io/example/TracerTest.java
@@ -10,7 +10,6 @@ class TracerTest {

     @Test
-    @DisabledOnOs(value = OS.WINDOWS, disabledReason = \"intermittent\")
     void startsTracer() {
         assertTrue(true);
     }
}
"""

    def test_extracts_reenabled_windows_junit_method(self):
        self.assertEqual(
            reenabled_windows_junit_test_suffixes(self.PATCH),
            ["TracerTest#startsTracer"],
        )

    def test_accepts_fully_qualified_expected_identity_and_ignores_other_platforms(self):
        instance = {
            "test_patch": self.PATCH,
            "FAIL_TO_PASS": ["io.example.TracerTest#startsTracer"],
        }
        self.assertEqual(
            windows_java_disabled_test_contract_is_gradable(instance, "windows"),
            (True, []),
        )
        self.assertEqual(
            windows_java_disabled_test_contract_is_gradable(
                {"test_patch": self.PATCH, "FAIL_TO_PASS": []}, "linux"
            ),
            (True, []),
        )

    def test_rejects_unrepresented_reenabled_windows_test(self):
        instance = {
            "test_patch": self.PATCH,
            "FAIL_TO_PASS": ["io.example.OtherTest#unrelated"],
        }
        self.assertEqual(
            windows_java_disabled_test_contract_is_gradable(instance, "windows"),
            (False, ["TracerTest#startsTracer"]),
        )

    def test_no_annotation_removal_is_not_blocked(self):
        unrelated_patch = self.PATCH.replace(
            '-    @DisabledOnOs(value = OS.WINDOWS, disabledReason = "intermittent")\n',
            '-    @DisabledOnOs(value = OS.LINUX, disabledReason = "intermittent")\n',
        )
        self.assertEqual(
            windows_java_disabled_test_contract_is_gradable(
                {"test_patch": unrelated_patch, "FAIL_TO_PASS": []}, "windows"
            ),
            (True, []),
        )

    def test_helidon9685_exact_published_contract_is_unavailable(self):
        helidon_patch = """diff --git a/microprofile/telemetry/src/test/java/io/helidon/microprofile/telemetry/TestTracerAtStartup.java b/microprofile/telemetry/src/test/java/io/helidon/microprofile/telemetry/TestTracerAtStartup.java
index f43cfa313e7..abc9a9f02bc 100644
--- a/microprofile/telemetry/src/test/java/io/helidon/microprofile/telemetry/TestTracerAtStartup.java
+++ b/microprofile/telemetry/src/test/java/io/helidon/microprofile/telemetry/TestTracerAtStartup.java
@@ -31,7 +31,6 @@ class TestTracerAtStartup {

     @Test
-    @DisabledOnOs(value = OS.WINDOWS, disabledReason = \"https://github.com/helidon-io/helidon/issues/9513\")
     void checkForFullFeaturedTracerAtStartup() {
"""
        self.assertEqual(
            windows_java_disabled_test_contract_is_gradable(
                {
                    "test_patch": helidon_patch,
                    "FAIL_TO_PASS": ["io.helidon.http.MediaTypeTest#testBuilt"],
                },
                "windows",
            ),
            (False, ["TestTracerAtStartup#checkForFullFeaturedTracerAtStartup"]),
        )

    def test_evaluation_is_unavailable_before_runtime_when_contract_is_missing(self):
        class RuntimeMustNotStart:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                raise AssertionError("runtime must not start for an unrepresented re-enabled test")

        original_runtime = evaluation.SetupRuntime
        evaluation.SetupRuntime = RuntimeMustNotStart
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                instance = {
                    "instance_id": "example__unrepresented-java-disabled-test",
                    "docker_image": "example-image",
                    "test_patch": self.PATCH,
                    "pred_patch": "solution-patch",
                    "PASS_TO_PASS": [],
                    "FAIL_TO_PASS": ["io.example.OtherTest#unrelated"],
                }
                report = evaluation.run_instance(instance, "windows", output_dir, overwrite=True)
                self.assertIsNone(report["resolved"])
                self.assertEqual(report["grading_status"], "unavailable")
                self.assertEqual(
                    report["failure_class"],
                    "unrepresented_windows_java_reenabled_test_regressions",
                )
                saved = json.loads(
                    (Path(output_dir) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved, report)
        finally:
            evaluation.SetupRuntime = original_runtime


if __name__ == "__main__":
    unittest.main()