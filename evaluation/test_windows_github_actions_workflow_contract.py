import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

if "datasets" not in sys.modules:
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: None
    sys.modules["datasets"] = datasets

# Evaluation imports the optional RepoLaunch submodule at import time. These
# unit tests substitute runtime execution and only need the evaluator module.
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

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "evaluation" / "evaluation.py"
SPEC = importlib.util.spec_from_file_location("workflow_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


OFFICIAL_WORKFLOW_PATH = ".github/workflows/e2e-main.yaml"
CANDIDATE_WORKFLOW_PATH = ".github/workflows/pr-check.yaml"
UNIT_COMMAND = (
    "powershell.exe -NoProfile -Command \""
    "pnpm test:unit -- --reporter=verbose"
    "\""
)


def patch_for(path: str) -> str:
    return "\n".join([
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        "@@ -1 +1 @@",
        "-old",
        "+new",
    ])


def instance() -> dict:
    return {
        "instance_id": "fixture__workflow_contract",
        "docker_image": "fixture:image",
        "rebuild_cmds": [],
        "test_cmds": [UNIT_COMMAND],
        "print_cmds": [],
        "test_patch": patch_for(OFFICIAL_WORKFLOW_PATH),
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": [],
        "pred_patch": patch_for(CANDIDATE_WORKFLOW_PATH),
    }


class WindowsGithubActionsWorkflowContractTests(unittest.TestCase):
    def test_detects_two_workflow_only_patches_with_unit_only_command(self) -> None:
        row = instance()
        evidence = EVALUATOR.windows_github_actions_workflow_contract_unobservable(
            row,
            row["pred_patch"],
            "windows",
        )
        self.assertEqual(
            evidence,
            {
                "official_workflow_paths": [OFFICIAL_WORKFLOW_PATH],
                "candidate_workflow_paths": [CANDIDATE_WORKFLOW_PATH],
                "published_test_command": UNIT_COMMAND,
            },
        )

    def test_requires_windows_workflow_only_patches_and_pnpm_unit_command(self) -> None:
        row = instance()
        self.assertIsNone(
            EVALUATOR.windows_github_actions_workflow_contract_unobservable(
                row,
                patch_for("packages/main/src/other.ts"),
                "windows",
            )
        )
        source_official = instance()
        source_official["test_patch"] = patch_for("packages/main/src/other.ts")
        self.assertIsNone(
            EVALUATOR.windows_github_actions_workflow_contract_unobservable(
                source_official,
                source_official["pred_patch"],
                "windows",
            )
        )
        non_unit = instance()
        non_unit["test_cmds"] = ["pnpm lint"]
        self.assertIsNone(
            EVALUATOR.windows_github_actions_workflow_contract_unobservable(
                non_unit,
                non_unit["pred_patch"],
                "windows",
            )
        )
        self.assertIsNone(
            EVALUATOR.windows_github_actions_workflow_contract_unobservable(
                row,
                row["pred_patch"],
                "linux",
            )
        )

    def test_explicit_workflow_validation_keeps_normal_scoring_path(self) -> None:
        for extra in (
            " ; actionlint .github/workflows/*.yaml",
            " ; gh workflow view pr-check.yaml",
        ):
            row = instance()
            row["test_cmds"] = [UNIT_COMMAND + extra]
            self.assertIsNone(
                EVALUATOR.windows_github_actions_workflow_contract_unobservable(
                    row,
                    row["pred_patch"],
                    "windows",
                )
            )

    def test_reading_workflow_text_does_not_count_as_semantic_validation(self) -> None:
        row = instance()
        row["test_cmds"] = [UNIT_COMMAND + " ; Get-Content .github/workflows/pr-check.yaml"]
        self.assertIsNotNone(
            EVALUATOR.windows_github_actions_workflow_contract_unobservable(
                row,
                row["pred_patch"],
                "windows",
            )
        )

    def test_run_instance_persists_unavailable_without_scoring_unit_results(self) -> None:
        row = instance()
        original = EVALUATOR.evaluate_instance

        def fake_evaluate(*_args, **_kwargs):
            return {
                "unrelated::renderer_unit": "fail",
                "unrelated::another_unit": "pass",
            }

        EVALUATOR.evaluate_instance = fake_evaluate
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = EVALUATOR.run_instance(row, "windows", directory, True)
                saved = json.loads(
                    (Path(directory) / row["instance_id"] / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
        finally:
            EVALUATOR.evaluate_instance = original
        self.assertEqual(saved, report)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(
            report["failure_class"],
            "windows_github_actions_workflow_contract_unobservable_by_published_unit_command",
        )


if __name__ == "__main__":
    unittest.main()