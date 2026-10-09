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

# The launch implementation is an optional Git submodule. These contract tests
# substitute runtime execution and only need the evaluator module to import.
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
SPEC = importlib.util.spec_from_file_location("snapshot_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


TARGET = "requirement::tests::read_wrong_version"
SNAPSHOT = "compiler-core/src/snapshots/gleam_core__requirement__tests__read_wrong_version.snap"
COMPLETED_CARGO_LOG = "test result: ok. 1 passed; 0 failed; 0 ignored"


def instance(test_patch: str | None = None) -> dict:
    return {
        "instance_id": "fixture__rust_snapshot_identity",
        "docker_image": "fixture:image",
        "rebuild_cmds": [],
        "test_cmds": ["cargo test -- --nocapture"],
        "print_cmds": [],
        "test_patch": test_patch or "\n".join([
            f"diff --git a/{SNAPSHOT} b/{SNAPSHOT}",
            f"--- a/{SNAPSHOT}",
            f"+++ b/{SNAPSHOT}",
            "@@ -1 +1 @@",
            "-old snapshot",
            "+new snapshot",
        ]),
        "PASS_TO_PASS": ["requirement::tests::read_requirement"],
        "FAIL_TO_PASS": [TARGET],
        "pred_patch": "diff --git a/lib.rs b/lib.rs\n",
    }


class RustSnapshotExpectedIdentityContractTests(unittest.TestCase):
    def test_detects_snapshot_only_missing_expected_identity(self) -> None:
        evidence = EVALUATOR.windows_rust_snapshot_expected_test_identity_unobservable(
            instance(),
            {"requirement::tests::read_requirement": "pass"},
            COMPLETED_CARGO_LOG,
            "windows",
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["missing_fail_to_pass"], [TARGET])
        self.assertEqual(evidence["changed_snapshot_paths"], [SNAPSHOT])
        self.assertEqual(
            evidence["snapshot_paths_by_expected_identity"][TARGET], [SNAPSHOT]
        )

    def test_does_not_trigger_when_published_identity_is_observed(self) -> None:
        self.assertIsNone(
            EVALUATOR.windows_rust_snapshot_expected_test_identity_unobservable(
                instance(),
                {TARGET: "pass"},
                COMPLETED_CARGO_LOG,
                "windows",
            )
        )

    def test_does_not_trigger_for_source_test_patch_or_linux(self) -> None:
        source_patch = "\n".join([
            "diff --git a/compiler-core/src/requirement.rs b/compiler-core/src/requirement.rs",
            "--- a/compiler-core/src/requirement.rs",
            "+++ b/compiler-core/src/requirement.rs",
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ])
        self.assertIsNone(
            EVALUATOR.windows_rust_snapshot_expected_test_identity_unobservable(
                instance(source_patch),
                {"requirement::tests::read_requirement": "pass"},
                COMPLETED_CARGO_LOG,
                "windows",
            )
        )
        self.assertIsNone(
            EVALUATOR.windows_rust_snapshot_expected_test_identity_unobservable(
                instance(),
                {"requirement::tests::read_requirement": "pass"},
                COMPLETED_CARGO_LOG,
                "linux",
            )
        )

    def test_incomplete_or_failed_cargo_log_does_not_trigger(self) -> None:
        for log in (
            "running 1 test",
            "test result: ok. 1 passed; 0 failed\ntest result: FAILED. 0 passed; 1 failed",
        ):
            self.assertIsNone(
                EVALUATOR.windows_rust_snapshot_expected_test_identity_unobservable(
                    instance(),
                    {"requirement::tests::read_requirement": "pass"},
                    log,
                    "windows",
                )
            )

    def test_run_instance_persists_unavailable_without_scoring(self) -> None:
        original = EVALUATOR.evaluate_instance

        def fake_evaluate(*args, **_kwargs):
            output_dir = Path(args[-1])
            (output_dir / "post_patch_log.txt").write_text(COMPLETED_CARGO_LOG, encoding="utf-8")
            return {"requirement::tests::read_requirement": "pass"}

        EVALUATOR.evaluate_instance = fake_evaluate
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = EVALUATOR.run_instance(instance(), "windows", directory, True)
                saved = json.loads(
                    (Path(directory) / "fixture__rust_snapshot_identity" / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
        finally:
            EVALUATOR.evaluate_instance = original
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(
            report["failure_class"],
            "windows_rust_snapshot_expected_test_identity_unobservable",
        )
        self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
