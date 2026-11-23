import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module_assignments(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, int):
                values[target.id] = node.value.value
    return values


def _command_timeout_keywords(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "command_timeout" and isinstance(keyword.value, ast.Name):
                values.append(keyword.value.id)
    return values


class TimeoutContractTests(unittest.TestCase):
    def test_evaluator_timeout_constants_are_minutes(self):
        evaluation = ROOT / "evaluation" / "evaluation.py"
        validation = ROOT / "evaluation" / "validation.py"

        self.assertEqual(_module_assignments(evaluation)["COMMAND_TIMEOUT_MINUTES"], 150)
        self.assertEqual(_module_assignments(validation)["VALIDATION_TIMEOUT_MINUTES"], 90)
        self.assertEqual(_command_timeout_keywords(evaluation), ["COMMAND_TIMEOUT_MINUTES"])
        self.assertEqual(
            _command_timeout_keywords(validation),
            ["VALIDATION_TIMEOUT_MINUTES", "VALIDATION_TIMEOUT_MINUTES"],
        )

    def test_legacy_seconds_as_minutes_contract_is_absent(self):
        for path in (
            ROOT / "evaluation" / "evaluation.py",
            ROOT / "evaluation" / "validation.py",
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("TIMEOUT = 150*60", source)
            self.assertNotIn("TIMEOUT = 90*60", source)
            self.assertNotIn("command_timeout=TIMEOUT", source)


if __name__ == "__main__":
    unittest.main()