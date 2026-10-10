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

from evaluation.windows_dotnet_test_patch_contract import (
    added_dotnet_test_methods,
    windows_dotnet_test_patch_contract_is_gradable,
)

evaluation = importlib.import_module("evaluation.evaluation")


class WindowsDotnetTestPatchContractTests(unittest.TestCase):
    PATCH = """diff --git a/tests/Avalonia.Base.UnitTests/Media/FontFamilyTests.cs b/tests/Avalonia.Base.UnitTests/Media/FontFamilyTests.cs
index 1111111..2222222 100644
--- a/tests/Avalonia.Base.UnitTests/Media/FontFamilyTests.cs
+++ b/tests/Avalonia.Base.UnitTests/Media/FontFamilyTests.cs
@@ -1,3 +1,20 @@
 namespace Avalonia.Base.UnitTests.Media;
+
+[Theory]
+[InlineData(null, "Arial", "Arial", null)]
+public void Should_Parse_FontFamily_With_BaseUri(string baseUri, string input, string expectedName, string expectedUri)
+{
+}
+
+[Theory]
+[InlineData("avares://MyAssembly/", "Some/Path/#FontName")]
+public void Should_Parse_Relative_Path(string baseUri, string path)
+{
+}
+diff --git a/tests/Avalonia.Skia.UnitTests/Media/FontManagerTests.cs b/tests/Avalonia.Skia.UnitTests/Media/FontManagerTests.cs
index 1111111..2222222 100644
--- a/tests/Avalonia.Skia.UnitTests/Media/FontManagerTests.cs
+++ b/tests/Avalonia.Skia.UnitTests/Media/FontManagerTests.cs
@@ -1,3 +1,10 @@
 namespace Avalonia.Skia.UnitTests.Media;
+
+[Win32Theory("Windows specific font")]
+public void Should_Get_SystemFont_With_BaseUri(string name)
+{
+}
+diff --git a/src/Avalonia.Base/Media/FontFamily.cs b/src/Avalonia.Base/Media/FontFamily.cs
index 1111111..2222222 100644
--- a/src/Avalonia.Base/Media/FontFamily.cs
+++ b/src/Avalonia.Base/Media/FontFamily.cs
@@ -1,2 +1,2 @@
-public string Old() => "old";
+public string New() => "new";
+"""

    def test_extracts_added_attributed_methods_from_test_sources(self):
        self.assertEqual(
            added_dotnet_test_methods(self.PATCH),
            [
                "Should_Parse_FontFamily_With_BaseUri",
                "Should_Parse_Relative_Path",
                "Should_Get_SystemFont_With_BaseUri",
            ],
        )

    def test_accepts_represented_method_and_ignores_non_windows_rows(self):
        instance = {
            "test_patch": self.PATCH,
            "FAIL_TO_PASS": [
                "Avalonia.Base.UnitTests.Media.FontFamilyTests.Should_Parse_Relative_Path",
            ],
        }
        self.assertEqual(
            windows_dotnet_test_patch_contract_is_gradable(instance, "windows"),
            (True, []),
        )
        self.assertEqual(
            windows_dotnet_test_patch_contract_is_gradable(
                {"test_patch": self.PATCH, "FAIL_TO_PASS": []}, "linux"
            ),
            (True, []),
        )

    def test_rejects_unrepresented_windows_dotnet_test_patch_methods(self):
        instance = {
            "test_patch": self.PATCH,
            "FAIL_TO_PASS": [
                "Avalonia.Skia.RenderTests.PathTests.Path_With_PenLineCap",
            ],
        }
        self.assertEqual(
            windows_dotnet_test_patch_contract_is_gradable(instance, "windows"),
            (
                False,
                [
                    "Should_Parse_FontFamily_With_BaseUri",
                    "Should_Parse_Relative_Path",
                    "Should_Get_SystemFont_With_BaseUri",
                ],
            ),
        )

    def test_ignores_unattributed_or_non_test_source_methods(self):
        patch = """diff --git a/tests/Project.Tests/Helper.cs b/tests/Project.Tests/Helper.cs
index 1111111..2222222 100644
--- a/tests/Project.Tests/Helper.cs
+++ b/tests/Project.Tests/Helper.cs
@@ -1,2 +1,5 @@
 public class Helper
 {
+    public void NotATest() {}
 }
+diff --git a/src/Project/FeatureTests.cs b/src/Project/FeatureTests.cs
+index 1111111..2222222 100644
+--- a/src/Project/FeatureTests.cs
++++ b/src/Project/FeatureTests.cs
+@@ -1,2 +1,6 @@
+public class FeatureTests
+{
++    [Fact]
++    public void AlsoNotATestSource() {}
+}
+"""
        self.assertEqual(added_dotnet_test_methods(patch), [])
        self.assertEqual(
            windows_dotnet_test_patch_contract_is_gradable(
                {"test_patch": patch, "FAIL_TO_PASS": []}, "windows"
            ),
            (True, []),
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
                    "instance_id": "AvaloniaUI__Avalonia-19538",
                    "docker_image": "example-image",
                    "test_patch": self.PATCH,
                    "pred_patch": "solution-patch",
                    "PASS_TO_PASS": [],
                    "FAIL_TO_PASS": ["Avalonia.Skia.RenderTests.PathTests.Path_With_PenLineCap"],
                }
                report = evaluation.run_instance(instance, "windows", output_dir, overwrite=True)
                self.assertIsNone(report["resolved"])
                self.assertEqual(report["grading_status"], "unavailable")
                self.assertEqual(
                    report["failure_class"],
                    "unrepresented_windows_dotnet_test_patch_regressions",
                )
                self.assertEqual(
                    report["added_test_methods"],
                    [
                        "Should_Parse_FontFamily_With_BaseUri",
                        "Should_Parse_Relative_Path",
                        "Should_Get_SystemFont_With_BaseUri",
                    ],
                )
                saved = json.loads(
                    (Path(output_dir) / instance["instance_id"] / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(saved, report)
        finally:
            evaluation.SetupRuntime = original_runtime


if __name__ == "__main__":
    unittest.main()