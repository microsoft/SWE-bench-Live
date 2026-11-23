from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_timeout_is_passed_in_minutes():
    source = (ROOT / "evaluation" / "evaluation.py").read_text(encoding="utf-8")

    assert "COMMAND_TIMEOUT_MINUTES = 150" in source
    assert "command_timeout=COMMAND_TIMEOUT_MINUTES" in source
    assert "TIMEOUT = 150*60" not in source


def test_validation_timeout_is_passed_in_minutes():
    source = (ROOT / "evaluation" / "validation.py").read_text(encoding="utf-8")

    assert "COMMAND_TIMEOUT_MINUTES = 90" in source
    assert "command_timeout=COMMAND_TIMEOUT_MINUTES" in source
    assert "TIMEOUT = 90*60" not in source