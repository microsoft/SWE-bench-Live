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


def _instance():
    actual_p2p = [f"Fixture.TestStable{index:02d}" for index in range(20)]
    expected_p2p = [name + "x" for name in actual_p2p]
    actual_f2p = ["Fixture.RegressionAlpha", "Fixture.RegressionBeta", "Fixture.RegressionGamma"]
    expected_f2p = ["Fixture.RegressionAlphaa", "Fixture.RegressionBetaa", "Fixture.RegressionGammaa"]
    return actual_p2p, expected_p2p, actual_f2p, expected_f2p


def _completed_trx(actual):
    results = "".join(
        f'<UnitTestResult testName="{name}" outcome="Passed" />' for name in actual
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<TestRun><Results>' + results + '</Results>'
        '<ResultSummary outcome="Completed"><Counters total="23" executed="23" passed="23" '
        'failed="0" error="0" timeout="0" aborted="0" /></ResultSummary></TestRun>'
    )


class _Container:
    def __init__(self, output):
        self.output = output
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, _command):
        return types.SimpleNamespace(output=self.output)

    def cleanup(self):
        self.cleaned = True


class WindowsDotnetTrxExpectedNameContractTests(unittest.TestCase):
    def test_detects_completed_trx_with_systematic_metadata_drift(self):
        actual_p2p, expected_p2p, actual_f2p, expected_f2p = _instance()
        status, info = evaluator.parse_windows_dotnet_trx_status(_completed_trx(actual_p2p + actual_f2p))
        evidence = evaluator.windows_dotnet_trx_expected_name_corruption(
            {"test_cmds": ["dotnet test Common.Tests.csproj"], "PASS_TO_PASS": expected_p2p, "FAIL_TO_PASS": expected_f2p},
            status,
            info,
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["exact_expected_match_count"], 0)
        self.assertEqual(evidence["unique_distance_one_passing_neighbor_count"], 23)
        self.assertEqual(
            evidence["affected_fail_to_pass"]["Fixture.RegressionAlphaa"]["observed"],
            "Fixture.RegressionAlpha",
        )

    def test_requires_completed_dotnet_trx_and_no_exact_expected_name(self):
        actual_p2p, expected_p2p, actual_f2p, expected_f2p = _instance()
        status, info = evaluator.parse_windows_dotnet_trx_status(_completed_trx(actual_p2p + actual_f2p))
        instance = {"test_cmds": ["dotnet test Common.Tests.csproj"], "PASS_TO_PASS": expected_p2p, "FAIL_TO_PASS": expected_f2p}
        self.assertIsNone(evaluator.windows_dotnet_trx_expected_name_corruption({**instance, "test_cmds": ["pytest -q"]}, status, info))
        self.assertIsNone(evaluator.windows_dotnet_trx_expected_name_corruption(instance, status, None))
        self.assertIsNone(evaluator.windows_dotnet_trx_expected_name_corruption({**instance, "PASS_TO_PASS": [actual_p2p[0]] + expected_p2p[1:]}, status, info))

    def test_run_instance_persists_unavailable_without_claiming_a_solution(self):
        actual_p2p, expected_p2p, actual_f2p, expected_f2p = _instance()
        instance = {
            "instance_id": "fixture__dotnet_trx_expected_names",
            "docker_image": "fixture:image",
            "rebuild_cmds": [],
            "test_cmds": ["dotnet test Common.Tests.csproj"],
            "print_cmds": ["print-command"],
            "test_patch": "test patch",
            "pred_patch": "solution patch",
            "PASS_TO_PASS": expected_p2p,
            "FAIL_TO_PASS": expected_f2p,
        }
        container = _Container(_completed_trx(actual_p2p + actual_f2p))
        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluator, "SetupRuntime", _Runtime), patch.object(evaluator, "run_parser", return_value={}):
                report = evaluator.run_instance(instance, "windows", directory, True)
            on_disk = json.loads((Path(directory) / instance["instance_id"] / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(container.cleaned)
        self.assertEqual(report, on_disk)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(report["failure_class"], "windows_dotnet_trx_expected_name_corruption")


if __name__ == "__main__":    unittest.main()
