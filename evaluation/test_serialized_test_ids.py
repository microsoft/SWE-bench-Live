import sys
import types


# Keep these evaluator-unit tests runnable without a colocated RepoLaunch checkout.
launch = types.ModuleType("launch")
launch.__path__ = []
launch_core = types.ModuleType("launch.core")
launch_scripts = types.ModuleType("launch.scripts")
runtime = types.ModuleType("launch.core.runtime")
runtime.SetupRuntime = object
parser = types.ModuleType("launch.scripts.parser")
parser.run_parser = lambda *_args, **_kwargs: {}
datasets = types.ModuleType("datasets")
datasets.load_dataset = lambda *_args, **_kwargs: None
sys.modules.setdefault("launch", launch)
sys.modules.setdefault("launch.core", launch_core)
sys.modules.setdefault("launch.scripts", launch_scripts)
sys.modules.setdefault("launch.core.runtime", runtime)
sys.modules.setdefault("launch.scripts.parser", parser)
sys.modules.setdefault("datasets", datasets)

from evaluation import evaluation


def test_normalize_test_id_sequence_decodes_native_and_serialized_values():
    assert evaluation.normalize_test_id_sequence(None) == []
    assert evaluation.normalize_test_id_sequence(["one", "two"]) == ["one", "two"]
    assert evaluation.normalize_test_id_sequence("tests/test_ha.py::TestHa::test_post_recover") == [
        "tests/test_ha.py::TestHa::test_post_recover"
    ]
    assert evaluation.normalize_test_id_sequence("['one', 'two']") == ["one", "two"]
    assert evaluation.normalize_test_id_sequence("['one'\n 'two']") == ["one", "two"]


def test_run_instance_scores_serialized_test_id_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "evaluate_instance",
        lambda *args, **kwargs: {"target": "pass", "stable": "pass"},
    )
    instance = {
        "instance_id": "synthetic__serialized-test-ids",
        "docker_image": "unused",
        "rebuild_cmds": [],
        "test_cmds": [],
        "print_cmds": [],
        "test_patch": "test patch",
        "pred_patch": "candidate patch",
        "log_parser": "pytest",
        "FAIL_TO_PASS": "['target']",
        "PASS_TO_PASS": "['stable']",
    }

    report = evaluation.run_instance(instance, "linux", str(tmp_path), overwrite=True)

    assert report["resolved"] is True
    assert report["FAIL_TO_PASS"] == {"success": ["target"], "failure": []}
    assert report["PASS_TO_PASS"] == {"success": ["stable"], "failure": []}
