from evaluation import evaluation


def _instance():
    return {
        "instance_id": "synthetic__gold-reference-unscorable",
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


def test_matching_gold_target_failure_is_unscorable(monkeypatch, tmp_path):
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

    assert report["resolved"] is None
    assert report["result_kind"] == "unscorable"
    assert report["gold_reference_control"] == {
        "performed": True,
        "matching_gold_reference_failures": ["target"],
        "candidate_only_failures": [],
    }


def test_candidate_only_target_failure_remains_unresolved(monkeypatch, tmp_path):
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
    assert report["result_kind"] == "unresolved"
    assert report["gold_reference_control"] == {
        "performed": True,
        "matching_gold_reference_failures": [],
        "candidate_only_failures": ["target"],
    }


def test_unavailable_gold_control_preserves_unresolved_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "evaluate_instance",
        lambda *args, **kwargs: {"target": "fail", "stable": "pass"},
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("reference unavailable")

    monkeypatch.setattr(evaluation, "evaluate_gold_reference_instance", unavailable)

    report = evaluation.run_instance(_instance(), "linux", str(tmp_path), overwrite=True)

    assert report["resolved"] is False
    assert report["result_kind"] == "unresolved"
    assert report["gold_reference_control"] == {
        "performed": False,
        "error": "RuntimeError: reference unavailable",
    }


def test_reference_replay_applies_test_then_gold_patch(monkeypatch, tmp_path):
    class Response:
        output = "structured test log"

    class Container:
        def __init__(self):
            self.applied = []
            self.commands = []
            self.cleaned = False

        def apply_patch(self, patch, verbose=False):
            self.applied.append((patch, verbose))
            return True

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
        "synthetic__gold-reference-unscorable",
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
    assert container.commands == [
        "cd /testbed",
        """[ -d .git ] || { g=$(find . -maxdepth 2 -mindepth 2 -type d -name .git -print -quit); [ -n "$g" ] && cd "${g%/.git}"; } ;""",
        "cd /testbed",
        "rebuild-command",
        "test-command",
        "print-command",
    ]
    assert container.cleaned is True
    assert (tmp_path / "gold_reference_post_patch_log.txt").read_text(encoding="utf-8") == "structured test log"


def test_existing_unscorable_report_is_reused_without_replay(monkeypatch, tmp_path):
    instance = _instance()
    report_dir = tmp_path / instance["instance_id"]
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        '{"instance_id": "synthetic__gold-reference-unscorable", "resolved": null, "result_kind": "unscorable"}',
        encoding="utf-8",
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("an existing unscorable report must not be replayed")

    monkeypatch.setattr(evaluation, "evaluate_instance", should_not_run)

    report = evaluation.run_instance(instance, "linux", str(tmp_path), overwrite=False)

    assert report["result_kind"] == "unscorable"
    assert report["resolved"] is None


def test_unscorable_results_are_not_counted_as_failures(monkeypatch, tmp_path):
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

    results = evaluation.run_instances([_instance()], "linux", 1, str(tmp_path), overwrite=True)

    assert results["success_ids"] == []
    assert results["failure_ids"] == []
    assert results["unscorable_ids"] == ["synthetic__gold-reference-unscorable"]
    assert results["unscorable"] == 1
    assert results["failure"] == 0