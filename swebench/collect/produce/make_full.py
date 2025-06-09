from __future__ import annotations
import argparse
import json
import os
from datetime import date
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect valid instances to create full dataset")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
    )
    parser.add_argument(
        "--output-file", 
        type=Path,
    )
    
    args = parser.parse_args()
    
    if args.output_file is None:
        today = date.today().isoformat()
        args.output_file = args.output_dir / f"full-{today}.jsonl"
    
    if not args.input_dir.is_dir():
        raise SystemExit(f"Expected directory {args.input_dir} not found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with args.output_file.open("w", encoding="utf-8") as outfile:
        for instance_path in args.input_dir.rglob("instance.json"):
            try:
                with instance_path.open(encoding="utf-8") as fp:
                    dct = json.load(fp)
            except (json.JSONDecodeError, OSError) as err:
                print(f"Skipping {instance_path}: {err}")
                continue

            if dct.get("FAIL_TO_PASS") and dct.get("PASS_TO_PASS"):
                json.dump(dct, outfile, ensure_ascii=False)
                outfile.write("\n")

    print(f"Collected records written to {args.output_file.resolve()}")

if __name__ == "__main__":
    main()

