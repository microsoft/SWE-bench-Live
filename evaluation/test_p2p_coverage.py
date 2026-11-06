"""Regression tests for PASS_TO_PASS coverage (issue #102)."""
import ast
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

SOURCE = Path(__file__).with_name("evaluation.py")


class TestP2PCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        funcs = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "run_instance"
        ]
        if len(funcs) != 1:
            raise RuntimeError("Expected exactly one run_instance")

        module = ast.fix_missing_locations(
            ast.Module(body=funcs, type_ignores=[])
        )
        cls.ns = {
            "os": os,
            "json": json,
            "Literal": Literal,
            "get_default_image_name": lambda *_: "unused",
        }
        exec(compile(module, str(SOURCE), "exec"), cls.ns)

    def test_p2p_coverage_and_legacy_skip(self):
        cases = [
            ("all pass", ["old"], {"new": "pass", "old": "pass"}, True),
            ("P2P missing", ["old"], {"new": "pass"}, False),
            ("P2P unrelated pass", ["old"],
             {"new": "pass", "other": "pass"}, False),
            ("P2P partially missing", ["old", "old2"],
             {"new": "pass", "old": "pass"}, False),
            ("P2P failed", ["old"],
             {"new": "pass", "old": "fail"}, False),
            ("P2P skipped", ["old"],
             {"new": "pass", "old": "skip"}, True),
            ("F2P failed", ["old"],
             {"new": "fail", "old": "pass"}, False),
            ("no expected P2P", [], {"new": "pass"}, True),
        ]

        for label, p2p, statuses, expected in cases:
            with self.subTest(label=label):
                self.ns["evaluate_instance"] = (
                    lambda *_, **__: dict(statuses)
                )
                instance = {
                    "instance_id": "offline",
                    "docker_image": "unused",
                    "test_patch": "",
                    "pred_patch": "",
                    "FAIL_TO_PASS": ["new"],
                    "PASS_TO_PASS": p2p,
                }
                with TemporaryDirectory() as tmp:
                    report = self.ns["run_instance"](
                        instance, "linux", tmp, True
                    )
                    saved = json.loads(
                        (Path(tmp) / "offline" / "report.json")
                        .read_text()
                    )
                self.assertEqual(report, saved)
                self.assertIs(report["resolved"], expected)


if __name__ == "__main__":
    unittest.main()
