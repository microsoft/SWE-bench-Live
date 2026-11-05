import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "launch"))
from launch.core.runtime import SetupRuntime
from launch.scripts.parser import run_parser
import json
import argparse
import traceback
import re
import xml.etree.ElementTree as ET
from typing import Iterable, Literal, TypedDict
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

TIMEOUT = 150*60


_DOTNET_TEST_COMMAND = re.compile(r"\bdotnet(?:\.exe)?\s+test\b", re.IGNORECASE)
_DOTNET_TRX_EXPECTED_MINIMUM = 20
_DOTNET_TRX_F2P_MINIMUM = 3
_DOTNET_TRX_ONE_EDIT_MINIMUM = 3
_DOTNET_TRX_ONE_EDIT_RATIO = 0.20
_DOTNET_TRX_F2P_MAX_DISTANCE = 3


def _trx_status(outcome: object) -> str:
    value = str(outcome or "").strip().lower()
    if value in {"passed", "pass", "success", "succeeded", "ok"}:
        return "pass"
    if value in {"skipped", "notexecuted", "not executed", "inconclusive", "ignored", "disabled"}:
        return "skip"
    return "fail"


def parse_windows_dotnet_trx_status(log: str) -> tuple[dict[str, str], dict | None]:
    """Read exact parameterized identities from a completed TRX capture.

    This is evidence extraction only. It deliberately preserves every
    UnitTestResult ``testName`` and does not translate the value to a published
    benchmark name.
    """
    text = str(log or "")
    start = text.rfind("<?xml")
    if start < 0:
        return {}, None
    end = text.find("</TestRun>", start)
    if end < 0:
        return {}, None
    try:
        root = ET.fromstring(text[start:end + len("</TestRun>")])
    except ET.ParseError:
        return {}, None
    if not str(root.tag).lower().endswith("testrun"):
        return {}, None
    summary = root.find(".//{*}ResultSummary")
    if summary is None or str(summary.attrib.get("outcome", "")).strip().lower() != "completed":
        return {}, None
    counters = summary.find(".//{*}Counters")
    status: dict[str, str] = {}
    precedence = {"pass": 0, "skip": 1, "fail": 2}
    for node in root.findall(".//{*}Results/{*}UnitTestResult"):
        name = str(node.attrib.get("testName", "") or "")
        if not name:
            continue
        outcome = _trx_status(node.attrib.get("outcome", ""))
        current = status.get(name)
        if current is None or precedence[outcome] > precedence[current]:
            status[name] = outcome
    if not status:
        return {}, None
    counter_values: dict[str, int] = {}
    if counters is not None:
        for key in ("total", "executed", "passed", "failed", "error", "timeout", "aborted"):
            try:
                counter_values[key] = int(counters.attrib.get(key, 0))
            except (TypeError, ValueError):
                counter_values[key] = 0
    return status, {
        "result_count": len(status),
        "result_summary_outcome": str(summary.attrib.get("outcome", "")),
        "counters": counter_values,
    }


def _unique_distance_one_passing_neighbors(expected: Iterable[str], passing: set[str]) -> dict[str, str]:
    """Return unambiguous literal one-edit passing neighbors as evidence only."""
    deleted_index: dict[str, set[str]] = {}
    substitution_index: dict[tuple[int, str], set[str]] = {}
    for actual in passing:
        for index in range(len(actual)):
            deleted = actual[:index] + actual[index + 1:]
            deleted_index.setdefault(deleted, set()).add(actual)
            substitution_index.setdefault((index, deleted), set()).add(actual)
    matches: dict[str, str] = {}
    for name in expected:
        if name in passing:
            continue
        candidates: set[str] = set(deleted_index.get(name, set()))
        for index in range(len(name)):
            shorter = name[:index] + name[index + 1:]
            if shorter in passing:
                candidates.add(shorter)
            candidates.update(substitution_index.get((index, shorter), set()))
        if len(candidates) == 1:
            matches[name] = next(iter(candidates))
    return matches


def _bounded_edit_distance(left: str, right: str, maximum: int) -> int:
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        row_minimum = row
        for column, right_char in enumerate(right, 1):
            value = min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return maximum + 1
        previous = current
    return previous[-1] if previous[-1] <= maximum else maximum + 1


def windows_dotnet_trx_expected_name_corruption(
    instance: dict,
    trx_status: dict[str, str],
    trx_info: dict | None,
) -> dict | None:
    """Preserve unavailable rows with systematic TRX/metadata identity drift.

    This is not fuzzy grading: it does not replace expected names or award a
    candidate pass. It requires a completed Windows ``dotnet test`` TRX report,
    no exact expected-name observation, a large systematic unique-one-edit
    passing pattern, and one unique close passing neighbor for every F2P name.
    """
    test_command = " ; ".join(instance.get("test_cmds", []) or [])
    if _DOTNET_TEST_COMMAND.search(test_command) is None or trx_info is None:
        return None
    p2p = [str(name) for name in instance.get("PASS_TO_PASS", []) or []]
    f2p = [str(name) for name in instance.get("FAIL_TO_PASS", []) or []]
    expected = list(dict.fromkeys(p2p + f2p))
    if len(expected) < _DOTNET_TRX_EXPECTED_MINIMUM or len(f2p) < _DOTNET_TRX_F2P_MINIMUM:
        return None
    if any(name in trx_status for name in expected):
        return None
    passing = {name for name, outcome in trx_status.items() if outcome == "pass"}
    if not passing:
        return None
    one_edit = _unique_distance_one_passing_neighbors(expected, passing)
    if (
        len(one_edit) < _DOTNET_TRX_ONE_EDIT_MINIMUM
        or len(one_edit) / len(expected) < _DOTNET_TRX_ONE_EDIT_RATIO
    ):
        return None
    f2p_neighbors: dict[str, dict] = {}
    for expected_name in f2p:
        minimum = _DOTNET_TRX_F2P_MAX_DISTANCE + 1
        nearest: list[str] = []
        for actual in passing:
            distance = _bounded_edit_distance(expected_name, actual, _DOTNET_TRX_F2P_MAX_DISTANCE)
            if distance < minimum:
                minimum = distance
                nearest = [actual]
            elif distance == minimum:
                nearest.append(actual)
        if minimum > _DOTNET_TRX_F2P_MAX_DISTANCE or len(nearest) != 1:
            return None
        f2p_neighbors[expected_name] = {
            "observed": nearest[0],
            "outcome": "pass",
            "edit_distance": minimum,
        }
    return {
        "expected_test_count": len(expected),
        "exact_expected_match_count": 0,
        "completed_trx": trx_info,
        "unique_distance_one_passing_neighbor_count": len(one_edit),
        "unique_distance_one_passing_neighbor_ratio": len(one_edit) / len(expected),
        "unique_distance_one_passing_neighbor_examples": [
            {"published": name, "observed": one_edit[name]}
            for name in sorted(one_edit)[:12]
        ],
        "affected_fail_to_pass": f2p_neighbors,
    }


def normalize_month(month: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"Invalid month format: {month}. Expected YYYY-MM.")
    return month


def get_instance_month(instance: dict) -> str | None:
    created_at = instance.get("created_at")
    if created_at is None:
        return None
    created_at_str = str(created_at).strip()
    if not created_at_str:
        return None
    match = re.match(r"^(\d{4}-\d{2})", created_at_str)
    if match is None:
        return None
    return match.group(1)


def parse_log_pytest(log: str, test_spec: "TestSpec") -> dict[str, str]:
    """
    Copied from SWE-bench/swebench/harness/constants/__init__.py
    """
    class TestStatus(Enum):
        FAILED = "FAILED"
        PASSED = "PASSED"
        SKIPPED = "SKIPPED"
        ERROR = "ERROR"
        XFAIL = "XFAIL"
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def default_pytest_parser(log: str) -> dict[str, str]:
    mapping = parse_log_pytest(log, None)
    for test in mapping.keys():
        # Pytest XFAIL means an expected failure remained expected. SWE-bench
        # grading treats XFAIL identically to PASSED for FAIL_TO_PASS and as
        # maintained for PASS_TO_PASS, so do not collapse it into fail here.
        if mapping[test].upper() == 'XFAIL':
            mapping[test] = 'pass'
        elif 'pass' in mapping[test].lower():
            mapping[test] = 'pass'
        elif 'skip' in mapping[test].lower():
            mapping[test] = 'skip'
        else:
            mapping[test] = 'fail'
    return mapping

def get_default_image_name(instance_id: str, platform: Literal["windows", "linux"]) -> str:
    if platform == "linux":
        med = "x86_64"
    else:
        med = "win"
    name = instance_id.replace("__", "_1776_").lower()
    image = f"starryzhang/sweb.eval.{med}.{name}"
    return image

def apply_solution_patch_best_effort(solution_patch: str, 
                                    container: SetupRuntime,
                                    platform: Literal["windows", "linux"]) -> None:
    '''
    For some 1% corner cases where /testbed/repo_name is the actual project
    '''
    if platform == "linux":
        container.send_command("cd /testbed")
        container.send_command("""[ -d .git ] || { g=$(find . -maxdepth 2 -mindepth 2 -type d -name .git -print -quit); [ -n "$g" ] && cd "${g%/.git}"; } ;""")
        container.apply_patch(solution_patch, verbose=True)
        container.send_command("cd /testbed")
    else:
        container.send_command(r"cd C:\testbed")
        container.send_command(r"""if (-not (Test-Path .git)) { $g = Get-ChildItem -Directory -Recurse -Depth 2 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '.git' } | Select-Object -First 1; if ($g) { Set-Location $g.Parent.FullName } };""")
        container.apply_patch(solution_patch, verbose=True)
        container.send_command(r"cd C:\testbed")
    return

def evaluate_instance(  
                    instance_id: str,
                    image: str, 
                    rebuild_cmd: str, 
                    test_cmd: str, 
                    print_cmd: str,
                    test_patch: str, 
                    solution_patch: str,
                    parser: str,
                    platform: Literal["windows", "linux"],
                    output_dir: str,
                    ) -> tuple[dict[str, Literal['pass', 'fail', 'skip']], dict[str, str], dict | None]:
    container: SetupRuntime = SetupRuntime.from_launch_image(image, instance_id, platform, command_timeout=TIMEOUT)
    container.apply_patch(test_patch)
    container.apply_patch(solution_patch, verbose=True)
    # Remember to rebuild after modifications to source codes !!!
    if rebuild_cmd.strip():
        container.send_command(rebuild_cmd)
    if not print_cmd.strip():
        # for backward compatibility with SWE-bench-Live/SWE-bench-Live (Python)
        container.send_command(f"cat > run_test.sh <<'CC_PROMPT'\n{test_cmd}\nCC_PROMPT\n")
        test_cmd = "bash run_test.sh > testlog.out 2>&1"
        print_cmd = "cat testlog.out"
    container.send_command(test_cmd)
    post_patch_log: str = container.send_command(print_cmd).output
    with open(os.path.join(output_dir, "post_patch_log.txt"), "w", encoding="utf-8") as f:
        f.write(post_patch_log)
    if parser.lower().strip() == "pytest":
        # for backward compatibility with SWE-bench-Live/SWE-bench-Live (Python)
        post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = default_pytest_parser(post_patch_log)
    else:
        post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = run_parser(parser, post_patch_log)
    dotnet_trx_status, dotnet_trx_info = parse_windows_dotnet_trx_status(post_patch_log) if platform == "windows" else ({}, None)
    container.cleanup()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(post_patch_status, f, indent = True)
    return post_patch_status, dotnet_trx_status, dotnet_trx_info

def run_instance(
                    instance: dict,
                    platform: Literal["windows","linux"],
                    output_dir: str,
                    overwrite: bool,
                    collect_only: bool = False,
                ):
    instance_output_dir = os.path.join(output_dir, instance["instance_id"])
    report_dir = os.path.join(instance_output_dir, "report.json")
    if (collect_only or not overwrite) and os.path.exists(report_dir):
        try:
            with open(report_dir, encoding="utf-8") as f:
                report = json.load(f)
                if report.get("resolved", None) is not None:
                    suc = "Success!" if report["resolved"] else "Failed..."
                    print(suc, "(Skipped)", instance["instance_id"], flush=True)
                    return report
        except:
            pass
    if collect_only:
        print("Incomplete...", instance["instance_id"], flush=True)
        return {"instance_id": instance["instance_id"], "resolved": None}
    os.makedirs(instance_output_dir, exist_ok=True)
    res, dotnet_trx_status, dotnet_trx_info = evaluate_instance(
            instance["instance_id"],
            instance.get("docker_image", get_default_image_name(instance["instance_id"], platform)),
            " ; ".join(instance.get("rebuild_cmds", [])),
            " ; ".join(instance.get("test_cmds", [])),
            " ; ".join(instance.get("print_cmds", [])),
            instance["test_patch"],
            instance["pred_patch"],
            instance.get("log_parser", instance.get("parser", "")),
            platform,
            instance_output_dir
    )
    if platform == "windows":
        corruption = windows_dotnet_trx_expected_name_corruption(instance, dotnet_trx_status, dotnet_trx_info)
        if corruption is not None:
            report = {
                "instance_id": instance["instance_id"],
                "resolved": None,
                "grading_status": "unavailable",
                "failure_class": "windows_dotnet_trx_expected_name_corruption",
                "windows_dotnet_trx_expected_name_corruption": corruption,
            }
            with open(report_dir, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=True)
            print("Unavailable...", instance["instance_id"], flush=True)
            return report
    suc = [test for test in res.keys() if 'pass' in res[test].lower()]
    fail = [test for test in res.keys() if 'fail' in res[test].lower()]
    report = {
        "instance_id": instance["instance_id"],
        "resolved": False,
        "PASS_TO_PASS": {
            "success": list(set(suc)&set(instance["PASS_TO_PASS"])),
            "failure": list(set(fail)&set(instance["PASS_TO_PASS"])),
        }, 
        "FAIL_TO_PASS": {
            "success": list(set(suc)&set(instance["FAIL_TO_PASS"])),
            "failure": list(set(fail)&set(instance["FAIL_TO_PASS"])),
        },
    }
    f2p = set(instance["FAIL_TO_PASS"]).issubset(set(report["FAIL_TO_PASS"]["success"])) \
        or (len(report["FAIL_TO_PASS"]["success"]) == len(instance["FAIL_TO_PASS"]))
    if (len(report["PASS_TO_PASS"]["failure"]) == 0) \
        and (len(report["FAIL_TO_PASS"]["failure"]) == 0) \
        and f2p:
        report["resolved"] = True
        print("Success!", instance["instance_id"], flush=True)
    else:
        print("Failed...", instance["instance_id"], flush=True)

    with open(report_dir, "w", encoding="utf-8") as f:
        json.dump(report, f, indent = True)
    return report

def run_instances(instances: list[dict[str, str]],
                    platform: Literal["windows", "linux"],
                    workers: int,
                    output_dir: str,
                    overwrite: bool,
                    collect_only: bool = False):
    todos = []
    empty_instance_ids = []
    for i in instances:
        if i["pred_patch"].strip():
            todos.append(i)
        else:
            empty_instance_ids.append(i["instance_id"])
            print("Empty patch...", i["instance_id"], flush=True)
    results = {
        "submitted": len(instances),
        "submitted_ids": [i["instance_id"] for i in instances],
        "empty_patch": len(empty_instance_ids),
        "empty_patch_ids": empty_instance_ids,
        "success_ids": [],
        "failure_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the executor
        future_to_instance = {
            executor.submit(run_instance, instance, platform, output_dir, overwrite, collect_only): instance
            for instance in todos
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                if result["resolved"] is None:
                    results["incomplete_ids"].append(instance["instance_id"])
                elif result["resolved"]:
                    results["success_ids"].append(instance["instance_id"])
                else:
                    results["failure_ids"].append(instance["instance_id"])
            except Exception as e:
                print(f"Error processing instance {instance['instance_id']}: {e} \n{traceback.format_exc()}")
                results["error_ids"].append(instance["instance_id"])
    results["success"] = len(results["success_ids"])
    results["failure"] = len(results["failure_ids"])
    results["error"] = len(results["error_ids"])
    results["incomplete"] = len(results["incomplete_ids"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent = True)
    return results

def main(
            dataset: str,
            patch_dir: str, 
            platform: Literal["windows", "linux"], 
            workers: int, 
            output_dir: str, 
            overwrite: int,
            split: str|None = None,
            instance_ids: list[str] | None = None,
            start_month: str | None = None,
            end_month: str | None = None,
            collect_only: bool = False,
        ):
    if patch_dir.strip() != "gold":
        with open(patch_dir, encoding="utf-8") as f:
            preds = json.load(f)
        print(f"Loaded {len(preds)} predictions.")
    else:
        print("Running Ground Truth Patches...")
    if os.path.exists(dataset) and dataset.endswith(".jsonl"):
        with open(dataset, encoding="utf-8") as f:
            instances = [json.loads(i) for i in f]
    elif split is not None:
        instances = load_dataset(dataset, split=split)
    else:
        instances = []
        ds = load_dataset(dataset)
        if hasattr(ds, "keys"):
            for key in ds.keys():
                instances.extend(ds[key])
        else:
            instances = ds
    all_months = []
    for instance in instances:
        month = get_instance_month(instance)
        if month is not None:
            all_months.append(month)
    if all_months:
        oldest_month = min(all_months)
        newest_month = max(all_months)
        start_month = normalize_month(start_month) if start_month is not None else oldest_month
        end_month = normalize_month(end_month) if end_month is not None else newest_month
        if start_month > end_month:
            raise ValueError(f"Invalid month range: start-month ({start_month}) is after end-month ({end_month}).")
        print(f"Month filter: {start_month} to {end_month} (dataset range: {oldest_month} to {newest_month})")
    elif start_month is not None or end_month is not None:
        raise ValueError("Month filtering requires 'created_at' field, but no parseable 'created_at' values were found.")
    if instance_ids is not None:
        print(f"Evaluating {instance_ids} ......")
    todos = []
    for idx in range(len(instances)):
        if instance_ids is not None and instances[idx]["instance_id"] not in instance_ids:
            continue
        # Create a proper dict copy since HuggingFace Dataset rows don't support in-place modification
        instance = dict(instances[idx])
        month = get_instance_month(instance)
        if all_months and month is not None and (month < start_month or month > end_month):
            continue
        if patch_dir.strip() != "gold" and instance["instance_id"] in preds.keys():
            instance["pred_patch"] = preds[instance["instance_id"]]["model_patch"]
            todos.append(instance)
        elif patch_dir.strip() == "gold":
            instance["pred_patch"] = instance["patch"]
            todos.append(instance)
    results = run_instances(todos, platform, workers, output_dir, overwrite != 0, collect_only)
    print("Submitted:", results["submitted"])
    print("Success:", results["success"])
    print("Failure:", results["failure"])
    print("Empty:", results["empty_patch"])
    print("Error:", results["error"])
    print("Incomplete:", results["incomplete"])
    if results["incomplete_ids"]:
        print("Incomplete IDs:", ", ".join(results["incomplete_ids"]))
    print("Evaluation ended successfully.")
    if patch_dir.strip().lower() == "gold":
        with open(os.path.join(output_dir, "gold_patch_evaluated_instances.jsonl"), "w", encoding="utf-8") as f:
            for instance in instances:
                if instance["instance_id"] in results["success_ids"]:
                    f.write(json.dumps(instance)+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench instances")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--patch_dir", type=str, required=True, help="Path to patch file or 'gold' for ground truth")
    parser.add_argument("--platform", type=str, choices=["windows", "linux"], required=True, help="Platform to run on")
    parser.add_argument("--workers", type=int, required=True, help="Number of worker threads")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--overwrite", type=int, required=True, help="Overwrite existing results (0 or 1)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use")
    parser.add_argument("--instance_ids", type=str, nargs="+", default=None, help="Specific instance IDs to evaluate")
    parser.add_argument("--start-month", type=str, default=None, help="Start month (inclusive) in YYYY-MM format")
    parser.add_argument("--end-month", type=str, default=None, help="End month (inclusive) in YYYY-MM format")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing results without running unfinished instances")

    args = parser.parse_args()
    
    main(
        dataset=args.dataset,
        patch_dir=args.patch_dir,
        platform=args.platform,
        workers=args.workers,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        split=args.split,
        instance_ids=args.instance_ids,
        start_month=args.start_month,
        end_month=args.end_month,
        collect_only=args.collect_only,
    )
