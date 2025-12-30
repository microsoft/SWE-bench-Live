import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "launch"))
from launch.core.runtime import SetupRuntime
from launch.scripts.parser import run_parser
import json
import argparse
import traceback
from typing import Literal, TypedDict
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

TIMEOUT = 40*60


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
        if 'pass' in mapping[test].lower():
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
                    ) -> dict[str, Literal['pass', 'fail', 'skip']]:
    container: SetupRuntime = SetupRuntime.from_launch_image(image, instance_id, platform)
    container.apply_patch(test_patch)
    apply_solution_patch_best_effort(solution_patch, container, platform)
    # Remember to rebuild after modifications to source codes !!!
    if rebuild_cmd.strip():
        container.send_command(rebuild_cmd, timeout=TIMEOUT)
    if not print_cmd.strip():
        # for backward compatibility with SWE-bench-Live/SWE-bench-Live (Python)
        container.send_command(f"cat > run_test.sh <<'CC_PROMPT'\n{test_cmd}\nCC_PROMPT\n")
        test_cmd = "bash run_test.sh > testlog.out 2>&1"
        print_cmd = "cat testlog.out"
    container.send_command(test_cmd, timeout=TIMEOUT)
    post_patch_log: str = container.send_command(print_cmd).output
    with open(os.path.join(output_dir, "post_patch_log.txt"), "w") as f:
        f.write(post_patch_log)
    if parser.lower().strip() == "pytest":
        # for backward compatibility with SWE-bench-Live/SWE-bench-Live (Python)
        post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = default_pytest_parser(post_patch_log)
    else:
        post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = run_parser(parser, post_patch_log)
    container.cleanup()
    with open(os.path.join(output_dir, "status.json"), "w") as f:
        json.dump(post_patch_status, f, indent = True)
    return post_patch_status

def run_instance(
                    instance: dict, 
                    platform: Literal["windows","linux"], 
                    output_dir: str, 
                    overwrite: bool
                ):
    instance_output_dir = os.path.join(output_dir, instance["instance_id"])
    report_dir = os.path.join(instance_output_dir, "report.json")
    if (not overwrite) and os.path.exists(report_dir):
        try:
            with open(report_dir) as f:
                report = json.load(f)
                if report.get("resolved", None) is not None:
                    suc = "Success!" if report["resolved"] else "Failed..."
                    print(suc, "(Skipped)", instance["instance_id"], flush=True)
                    return report
        except:
            pass
    os.makedirs(instance_output_dir, exist_ok=True)
    res: dict[str, Literal['pass', 'fail', 'skip']] = evaluate_instance(
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
    if (len(report["PASS_TO_PASS"]["failure"]) == 0) \
        and (len(report["FAIL_TO_PASS"]["failure"]) == 0) \
        and (len(report["FAIL_TO_PASS"]["success"]) > 0) :
        report["resolved"] = True
        print("Success!", instance["instance_id"], flush=True)
    else:
        print("Failed...", instance["instance_id"], flush=True)

    with open(report_dir, "w") as f:
        json.dump(report, f, indent = True)
    return report

def run_instances(instances: list[dict[str, str]], 
                    platform: Literal["windows", "linux"], 
                    workers: int,
                    output_dir: str,
                    overwrite: bool):
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
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the executor
        future_to_instance = {
            executor.submit(run_instance, instance, platform, output_dir, overwrite): instance
            for instance in todos
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                if result["resolved"]:
                    results["success_ids"].append(instance["instance_id"])
                else:
                    results["failure_ids"].append(instance["instance_id"])
            except Exception as e:
                print(f"Error processing instance {instance['instance_id']}: {e} \n{traceback.format_exc()}")
                results["error_ids"].append(instance["instance_id"])
    results["success"] = len(results["success_ids"])
    results["failure"] = len(results["failure_ids"])
    results["error"] = len(results["error_ids"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
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
        ):
    if patch_dir.strip() != "gold":
        with open(patch_dir) as f:
            preds = json.load(f)
        print(f"Loaded {len(preds)} predictions.")
    else:
        print("Running Ground Truth Patches...")
    if os.path.exists(dataset) and dataset.endswith(".jsonl"):
        with open(dataset) as f:
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
    if instance_ids is not None:
        print(f"Evaluating {instance_ids} ......")
    todos = []
    for idx in range(len(instances)):
        if instance_ids is not None and instances[idx]["instance_id"] not in instance_ids:
            continue
        if patch_dir.strip() != "gold" and instances[idx]["instance_id"] in preds.keys():
            instances[idx]["pred_patch"] = preds[instances[idx]["instance_id"]]["model_patch"]
            todos.append(instances[idx])
        if patch_dir.strip() == "gold":
            instances[idx]["pred_patch"] = instances[idx]["patch"]
            todos.append(instances[idx])
    results = run_instances(todos, platform, workers, output_dir, overwrite != 0)
    print("Submitted:", results["submitted"])
    print("Success:", results["success"])
    print("Failure:", results["failure"])
    print("Empty:", results["empty_patch"])
    print("Error:", results["error"])
    print("Evaluation ended successfully.")

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
    
    args = parser.parse_args()
    
    main(
        dataset=args.dataset,
        patch_dir=args.patch_dir,
        platform=args.platform,
        workers=args.workers,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        split=args.split,
        instance_ids=args.instance_ids
    )

