import json
from pathlib import Path
from fire import Fire

def main(
    playground,
    output_jsonl,
):
    playground = Path(playground)
    swe_instances = []
    for subfolder in playground.iterdir():
        if not subfolder.is_dir():
            continue

        instance_path = subfolder / "instance.json"
        result_path = subfolder / "result.json"

        if not instance_path.exists() or not result_path.exists():
            continue

        instance = json.loads(instance_path.read_text())
        result = json.loads(result_path.read_text())

        if not result["completed"]:
            continue

        swe_instance = {
            **instance,
            "test_cmds": result["test_commands"],
            "log_parser": result.get("log_parser", "pytest"),
        }

        swe_instances.append(swe_instance)

    with open(output_jsonl, "w") as f:
        for instance in swe_instances:
            f.write(json.dumps(instance) + "\n")
    print(f"Saved {len(swe_instances)} instances to {output_jsonl}")

if __name__ == "__main__":
    Fire(main)
