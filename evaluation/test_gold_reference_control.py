from evaluation import evaluation


def _instance():
    return {
        "instance_id": "synthetic__gold-reference-control",
        "docker_image": "unused",
        "rebuild_cmds": [],
        "test_cmds": ["test-command"],
        "print_cmds": ["print-command"],
        "test_patch": "test patch",
        "patch": "gold patch",
        "pred_patch": "candidate patch",
        "log_parser": "pytest",
        "FAIL_TO_PASS": ["target"],
        "PASS_TO_PASS": ["stable"],
    }


def test_failed_target_is_compared_to_the_gold_reference(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "evaluate_instance",
        lambda *args, **kwargs: {"target": "fail", "stable": "pass"},
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_gold_reference_instance",
        lambda *args, **kwargs: {"target": "fail", "stable": "pass"},
    )

    report = evaluation.run_instance(_instance(), "linux", str(tmp_path), overwrite=True)

    assert report["resolved"] is False
    assert report["gold_reference_control"] == {
        "performed": True,
        "matching_gold_reference_failures": ["target"],
        "candidate_only_failures": [],
    }


def test_candidate_only_target_failure_remains_distinct(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "evaluate_instance",
        lambda *args, **kwargs: {"target": "fail", "stable": "pass"},
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_gold_reference_instance",
        lambda *args, **kwargs: {"target": "pass", "stable": "pass"},
    )

    report = evaluation.run_instance(_instance(), "linux", str(tmp_path), overwrite=True)

    assert report["resolved"] is False
    assert report["gold_reference_control"] == {
        "performed": True,
        "matching_gold_reference_failures": [],
        "candidate_only_failures": ["target"],
    }


def test_passing_candidate_does_not_start_gold_reference_control(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "evaluate_instance",
        lambda *args, **kwargs: {"target": "pass", "stable": "pass"},
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("gold-reference control must only run after target failure")

    monkeypatch.setattr(evaluation, "evaluate_gold_reference_instance", should_not_run)

    report = evaluation.run_instance(_instance(), "linux", str(tmp_path), overwrite=True)

    assert report["resolved"] is True
    assert "gold_reference_control" not in report


def test_gold_reference_replays_test_then_gold_patch(monkeypatch, tmp_path):
    class Response:
        output = "structured test log"

    class Container:
        def __init__(self):
            self.applied = []
            self.commands = []
            self.cleaned = False

        def apply_patch(self, patch, verbose=False):
            self.applied.append((patch, verbose))

        def send_command(self, command):
            self.commands.append(command)
            return Response()

        def cleanup(self):
            self.cleaned = True

    container = Container()

    class Runtime:
        @staticmethod
        def from_launch_image(*args, **kwargs):
            return container

    monkeypatch.setattr(evaluation, "SetupRuntime", Runtime)
    monkeypatch.setattr(evaluation, "run_parser", lambda parser, log: {"target": "fail"})

    status = evaluation.evaluate_gold_reference_instance(
        "synthetic__gold-reference-control",
        "unused",
        "rebuild-command",
        "test-command",
        "print-command",
        "test patch",
        "gold patch",
        "go",
        "linux",
        str(tmp_path),
    )

    assert status == {"target": "fail"}
    assert container.applied == [("test patch", True), ("gold patch", True)]
    assert container.commands == ["rebuild-command", "test-command", "print-command"]
    assert container.cleaned is True
    assert (tmp_path / "gold_reference_post_patch_log.txt").read_text(encoding="utf-8") == "structured test log"