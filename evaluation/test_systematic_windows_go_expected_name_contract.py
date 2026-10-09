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

# Evaluation imports the launch submodule at import time. These unit tests only
# exercise pure contract logic and substitute SetupRuntime where needed, so a
# minimal module stub keeps them independent of the optional Git submodule.
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


class _Container:
    def __init__(self, output: str):
        self.output = output
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, _command: str):
        return types.SimpleNamespace(output=self.output)

    def cleanup(self):
        self.cleaned = True


def _systematic_instance():
    return {
        "instance_id": "fixture__systematic_expected_names",
        "test_cmds": ["go test -json -v ./..."],
        "PASS_TO_PASS": [
            "TestAlphaa",
            "TestBetaa",
            "TestGammaa",
            "TestStable01",
            "TestStable02",
            "TestStable03",
            "TestStable04",
            "TestStable05",
            "TestStable06",
            "TestStable07",
            "TestStable08",
            "TestStable09",
            "TestStable10",
            "TestStable11",
            "TestStable12",
            "TestStable13",
            "TestStable14",
            "TestStable15",
            "TestStable16",
            "TestStable17",
        ],
        "FAIL_TO_PASS": ["TestDeltaa", "TestEpsilonn", "TestZetaa"],
    }


def _systematic_status():
    status = {
        "pkg::TestAlpha": "pass",
        "pkg::TestBeta": "pass",
        "pkg::TestGamma": "pass",
        "pkg::TestDelta": "pass",
        "pkg::TestEpsilon": "pass",
        "pkg::TestZeta": "pass",
    }
    for index in range(1, 18):
        status[f"pkg::TestStable{index:02d}"] = "pass"
    return status


class SystematicWindowsGoExpectedNameContractTests(unittest.TestCase):
    def test_detects_many_unique_distance_one_passes_including_every_f2p(self):
        evidence = evaluator.systematic_windows_go_expected_name_corruption(
            _systematic_instance(),
            _systematic_status(),
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["affected_fail_to_pass"], {
            "TestDeltaa": "pkg::TestDelta",
            "TestEpsilonn": "pkg::TestEpsilon",
            "TestZetaa": "pkg::TestZeta",
        })
        self.assertEqual(evidence["unique_distance_one_passing_neighbor_count"], 6)

    def test_requires_every_fail_to_pass_to_share_the_signature(self):
        instance = _systematic_instance()
        instance["FAIL_TO_PASS"] = instance["FAIL_TO_PASS"] + ["TestNotObserved"]
        self.assertIsNone(
            evaluator.systematic_windows_go_expected_name_corruption(
                instance,
                _systematic_status(),
            )
        )

    def test_rejects_small_or_non_go_rows(self):
        small = {
            "test_cmds": ["go test ./..."],
            "PASS_TO_PASS": ["TestAlphaa", "TestBetaa"],
            "FAIL_TO_PASS": ["TestDeltaa", "TestEpsilonn", "TestZetaa"],
        }
        self.assertIsNone(
            evaluator.systematic_windows_go_expected_name_corruption(
                small,
                _systematic_status(),
            )
        )
        non_go = _systematic_instance()
        non_go["test_cmds"] = ["pytest -q"]
        self.assertIsNone(
            evaluator.systematic_windows_go_expected_name_corruption(
                non_go,
                _systematic_status(),
            )
        )

    def test_ambiguous_passing_neighbors_fail_closed(self):
        status = _systematic_status()
        status["other::TestDeltas"] = "pass"
        self.assertIsNone(
            evaluator.systematic_windows_go_expected_name_corruption(
                _systematic_instance(),
                status,
            )
        )

    def test_ignores_synthesized_bare_expected_names_after_nonzero_scoring(self):
        status = _systematic_status()
        for name in _systematic_instance()["PASS_TO_PASS"] + _systematic_instance()["FAIL_TO_PASS"]:
            status[name] = "fail"
        evidence = evaluator.systematic_windows_go_expected_name_corruption(
            _systematic_instance(),
            status,
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["unique_distance_one_passing_neighbor_count"], 6)

    def test_supports_legacy_bare_raw_parser_identities(self):
        status = {
            name.rsplit("::", 1)[-1]: outcome
            for name, outcome in _systematic_status().items()
        }
        evidence = evaluator.systematic_windows_go_expected_name_corruption(
            _systematic_instance(),
            status,
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["unique_distance_one_passing_neighbor_count"], 6)

    def test_run_instance_persists_unavailable_without_claiming_a_solution(self):
        instance = _systematic_instance()
        instance.update({
            "docker_image": "fixture:image",
            "rebuild_cmds": [],
            "print_cmds": ["print-command"],
            "test_patch": "test patch",
            "pred_patch": "solution patch",
        })
        container = _Container("report")
        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluator, "SetupRuntime", _Runtime), patch.object(
                evaluator,
                "run_parser",
                return_value=_systematic_status(),
            ):
                report = evaluator.run_instance(instance, "windows", directory, True)
            on_disk = json.loads(
                (Path(directory) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
            )
        self.assertTrue(container.cleaned)
        self.assertEqual(report, on_disk)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(
            report["failure_class"],
            "systematic_windows_go_expected_name_corruption",
        )


if __name__ == "__main__":
    unittest.main()