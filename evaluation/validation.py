import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "launch"))
from launch.core.runtime import SetupRuntime
from launch.scripts.parser import run_parser
import json
from typing import Literal, TypedDict
from fire import Fire
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT = 90*60

class ExecutionResult(TypedDict):
    instance_id: str
    pre_patch_status: dict[str, Literal['pass', 'fail', 'skip']]
    post_patch_status: dict[str, Literal['pass', 'fail', 'skip']]

class ValidationResult(ExecutionResult):
    PASS_TO_PASS: list[str]
    FAIL_TO_PASS: list[str]

def compare(execution_res: ExecutionResult) -> ValidationResult:
    pre_pass = set()
    post_pass = set()
    for test in execution_res["pre_patch_status"].keys():
        assert execution_res["pre_patch_status"][test].lower() in {'pass', 'fail', 'skip'}
        if execution_res["pre_patch_status"][test].lower() == 'pass':
            pre_pass.add(test)
    for test in execution_res["post_patch_status"].keys():
        if execution_res["post_patch_status"][test].lower() == 'pass':
            post_pass.add(test)
    return {
        **execution_res,
        "PASS_TO_PASS": list(pre_pass&post_pass),
        "FAIL_TO_PASS": list(post_pass-pre_pass)
    }

def validate_instance(  
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
                    ) -> ValidationResult:
    container: SetupRuntime = SetupRuntime.from_launch_image(image, instance_id, platform, command_timeout=TIMEOUT)
    container.apply_patch(test_patch, verbose=True)
    # Remember to rebuild after modifications to source codes !!!
    container.send_command(rebuild_cmd)
    container.send_command(test_cmd)
    pre_patch_log: str = container.send_command(print_cmd).output
    with open(os.path.join(output_dir, "pre_patch_log.txt"), "w", encoding="utf-8") as f:
        f.write(pre_patch_log)
    pre_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = run_parser(parser, pre_patch_log)
    container.cleanup()
    del container

    container: SetupRuntime = SetupRuntime.from_launch_image(image, instance_id, platform, command_timeout=TIMEOUT)
    container.apply_patch(test_patch, verbose=True)
    container.apply_patch(solution_patch, verbose=True)
    container.send_command(rebuild_cmd)
    post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = {}
    post_patch_status_under_inspect: dict[int, dict[str, Literal['pass', 'fail', 'skip']]] = {}
    post_patch_log_accumulate: str = ""
    # 3 validation for stable states
    for check in range(3):
        container.send_command(test_cmd)
        post_patch_log: str = container.send_command(print_cmd).output
        post_patch_log_accumulate += f"eval No.{check} \n\n========  \n\n{post_patch_log} \n\n"
        post_patch_status_under_inspect[check] = run_parser(parser, post_patch_log)
    all_tests = set(post_patch_status_under_inspect[0].keys()) | set(post_patch_status_under_inspect[1].keys()) | set(post_patch_status_under_inspect[2].keys())
    for test in all_tests:
        all_status = [
            post_patch_status_under_inspect[0].get(test, "skip").lower(),
            post_patch_status_under_inspect[1].get(test, "skip").lower(),
            post_patch_status_under_inspect[2].get(test, "skip").lower(),
        ]
        assert all_status[0] in {'pass', 'fail', 'skip'}
        assert all_status[1] in {'pass', 'fail', 'skip'}
        assert all_status[2] in {'pass', 'fail', 'skip'}
        if 'fail' in all_status:
            post_patch_status[test] = 'fail'
        elif 'skip' in all_status:
            post_patch_status[test] = 'skip'
        else:
            post_patch_status[test] = 'pass'
    with open(os.path.join(output_dir, "post_patch_log.txt"), "w", encoding="utf-8") as f:
        f.write(post_patch_log_accumulate)
    container.cleanup()
    
    res: ValidationResult = compare({
        "instance_id": instance_id,
        "pre_patch_status": pre_patch_status,
        "post_patch_status": post_patch_status,
    })
    print(instance_id, ": num of fail_to_pass:", len(res["FAIL_TO_PASS"]), flush=True)
    if len(res["FAIL_TO_PASS"]) > 0:
        print("Find one valid instance:", instance_id, flush=True)
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent = True)
    return res

def run_instance(instance: dict[str, str], 
                    platform: Literal["windows", "linux"],
                    output_dir: str,
                    overwrite: bool) -> ValidationResult:
    instance_output_dir = os.path.join(output_dir, instance["instance_id"])
    if (not overwrite) and os.path.exists(os.path.join(instance_output_dir, "status.json")):
        with open(os.path.join(instance_output_dir, "status.json"), encoding="utf-8") as f:
            try:
                report = json.load(f)
                print("Skipping", instance["instance_id"], "num of fail_to_pass:", len(report["FAIL_TO_PASS"]), flush=True)
                return report
            except Exception as e:
                print(e, flush=True)
                pass
    os.makedirs(instance_output_dir, exist_ok=True)
    return {"FAIL_TO_PASS": []}

def run_instances(instances: list[dict[str, str]], 
                    platform: Literal["windows", "linux"], 
                    workers: int,
                    output_dir: str,
                    overwrite: bool) -> list[dict[str, str]]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the executor
        future_to_instance = {
            executor.submit(run_instance, instance, platform, output_dir, overwrite): instance
            for instance in instances
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                results.append({
                    **instance,
                    **result
                })
            except Exception as e:
                print(f"Error processing instance {instance['instance_id']}: {e}", flush = True)
    return results


def main(
            input_dir: str, 
            platform: Literal["windows", "linux"], 
            workers: int, 
            output_dir: str, 
            overwrite: int,
        ):
    with open(input_dir, encoding="utf-8") as f:
        instances = [json.loads(i) for i in f]
    print(f"Loaded {len(instances)} instances.")
    results = run_instances(instances, platform, workers, output_dir, overwrite != 0)
    filtered_res = [i for i in results if len(i["FAIL_TO_PASS"]) > 0]
    print(f"Saved {len(filtered_res)} valid instances.")
    with open(os.path.join(output_dir, "validated_instances.jsonl"), "w", encoding="utf-8") as f:
        for i in filtered_res:
            f.write(json.dumps(i)+"\n")

if __name__ == "__main__":
    Fire(main)
