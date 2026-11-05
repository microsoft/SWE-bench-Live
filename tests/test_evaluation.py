import ast
import pathlib
import re
import unittest


EVALUATION_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "evaluation" / "evaluation.py"
)


def load_normalizer():
    source = EVALUATION_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_test_identifier"
    )
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(EVALUATION_SOURCE), "exec"),
        namespace,
    )
    return namespace["normalize_test_identifier"]


class NormalizeTestIdentifierTests(unittest.TestCase):
    def setUp(self):
        self.normalize_test_identifier = load_normalizer()

    def test_removes_terminal_jest_duration(self):
        self.assertEqual(
            self.normalize_test_identifier("worker succeeds (17 ms)"),
            "worker succeeds",
        )
        self.assertEqual(
            self.normalize_test_identifier("worker succeeds (0 ms)"),
            "worker succeeds",
        )

    def test_preserves_non_duration_identifiers(self):
        self.assertEqual(
            self.normalize_test_identifier("worker succeeds"),
            "worker succeeds",
        )
        self.assertEqual(
            self.normalize_test_identifier("worker succeeds (17 s)"),
            "worker succeeds (17 s)",
        )
        self.assertEqual(
            self.normalize_test_identifier("worker succeeds (17 ms) extra"),
            "worker succeeds (17 ms) extra",
        )

    def test_aligns_expected_and_parsed_test_names(self):
        parsed_passes = {
            self.normalize_test_identifier("suite worker succeeds (24 ms)")
        }
        expected_fail_to_pass = {
            self.normalize_test_identifier("suite worker succeeds")
        }

        self.assertEqual(parsed_passes, expected_fail_to_pass)


if __name__ == "__main__":
    unittest.main()