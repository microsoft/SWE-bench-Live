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
SPEC = importlib.util.spec_from_file_location("vscode_extension_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)

EXTENSION = "terminal-suggest"
OFFICIAL_PATHS = [
    "extensions/terminal-suggest/src/test/completions/code.test.ts",
    "extensions/terminal-suggest/src/test/terminalSuggestMain.test.ts",
]
CANDIDATE_PATH = "extensions/terminal-suggest/src/terminalSuggestMain.ts"
CORE_COMMAND = (
    "node .\\node_modules\\mocha\\bin\\mocha.js "
    "test/unit/node/index.js --delay --ui=tdd"
)


def patch_for(paths: list[str]) -> str:
    chunks = []
    for path in paths:
        chunks.extend([
            f"diff --git a/{path} b/{path}",
            f"--- a/{path}",
            f"+++ b/{path}",
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ])
    return "\n".join(chunks)


def instance() -> dict:
    return {
        "instance_id": "fixture__vscode_extension_contract",
        "docker_image": "fixture:image",
        "rebuild_cmds": [],
        "test_cmds": [CORE_COMMAND],
        "print_cmds": [],
        "test_patch": patch_for(OFFICIAL_PATHS),
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": ["Editor Model / unrelated regression"],
        "pred_patch": patch_for([CANDIDATE_PATH]),
    }


class WindowsVSCodeExtensionContractTests(unittest.TestCase):
    def test_detects_extension_tests_unseen_by_published_core_runner(self) -> None:
        row = instance()
        evidence = EVALUATOR.windows_vscode_extension_test_contract_unobservable(
            row,
            row["pred_patch"],
            "windows",
            {"Editor Model / unrelated regression": "pass"},
            "  100 passing\n",
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["extension_root"], EXTENSION)
        self.assertEqual(evidence["official_extension_test_paths"], OFFICIAL_PATHS)
        self.assertEqual(evidence["candidate_extension_source_paths"], [CANDIDATE_PATH])

    def test_requires_windows_core_runner_extension_only_patches_and_absent_marker(self) -> None:
        row = instance()
        status = {"Editor Model / unrelated regression": "pass"}
        self.assertIsNone(
            EVALUATOR.windows_vscode_extension_test_contract_unobservable(
                row, row["pred_patch"], "linux", status, "100 passing"
            )
        )
        runner_targets_extension = instance()
        runner_targets_extension["test_cmds"] = [CORE_COMMAND + " ; node extensions/terminal-suggest/out/test.js"]
        self.assertIsNone(
            EVALUATOR.windows_vscode_extension_test_contract_unobservable(
                runner_targets_extension, runner_targets_extension["pred_patch"], "windows", status, "100 passing"
            )
        )
        test_output_mentions_extension = instance()
        self.assertIsNone(
            EVALUATOR.windows_vscode_extension_test_contract_unobservable(
                test_output_mentions_extension, test_output_mentions_extension["pred_patch"], "windows", status,
                "Terminal Suggest\n  5 passing"
            )
        )
        candidate_changes_test = instance()
        candidate_changes_test["pred_patch"] = patch_for([
            "extensions/terminal-suggest/src/test/terminalSuggestMain.test.ts"
        ])
        self.assertIsNone(
            EVALUATOR.windows_vscode_extension_test_contract_unobservable(
                candidate_changes_test, candidate_changes_test["pred_patch"], "windows", status, "100 passing"
            )
        )
        mixed_official = instance()
        mixed_official["test_patch"] = patch_for(OFFICIAL_PATHS + ["src/vs/base/test/common/foo.test.ts"])
        self.assertIsNone(
            EVALUATOR.windows_vscode_extension_test_contract_unobservable(
                mixed_official, mixed_official["pred_patch"], "windows", status, "100 passing"
            )
        )
        represented_expected = instance()
        represented_expected["FAIL_TO_PASS"] = ["Terminal Suggest / code.cmd"]
        self.assertIsNone(
            EVALUATOR.windows_vscode_extension_test_contract_unobservable(
                represented_expected, represented_expected["pred_patch"], "windows", status, "100 passing"
            )
        )

    def test_run_instance_persists_unavailable_without_scoring_core_results(self) -> None:
        row = instance()
        original = EVALUATOR.evaluate_instance

        def fake_evaluate(*args, **_kwargs):
            output_dir = Path(args[-1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "post_patch_log.txt").write_text("100 passing\n3 failing\n", encoding="utf-8")
            return {"Editor Model / unrelated regression": "fail"}

        EVALUATOR.evaluate_instance = fake_evaluate
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = EVALUATOR.run_instance(row, "windows", directory, True)
                saved = json.loads(
                    (Path(directory) / row["instance_id"] / "report.json").read_text(encoding="utf-8")
                )
        finally:
            EVALUATOR.evaluate_instance = original
        self.assertEqual(saved, report)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(
            report["failure_class"],
            "windows_vscode_extension_test_contract_unobservable_by_published_core_unit_command",
        )


if __name__ == "__main__":
    unittest.main()
