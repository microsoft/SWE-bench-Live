import json
from pathlib import Path
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets

parser = argparse.ArgumentParser(description="Collect valid instances to create full dataset")
parser.add_argument(
        "--input-dir", 
        type=str, 
    )
args = parser.parse_args()

new_data = Dataset.from_json(args.input_dir)
print(f"Load {len(new_data):,} lines of new data.")
out_path = Path(args.input_dir)
ds = load_dataset("SWE-bench-Live/SWE-bench-Live")["full"]
print(f"Load {len(ds):,} lines of old data.")
merged_ds = concatenate_datasets([ds, new_data])

with out_path.open("w", encoding="utf-8") as f:
    for row in merged_ds:                          # ← streams one row at a time; no full copy in RAM
        row = dict(row)                            # Dataset row → plain dict
        # Format the timestamp exactly like 2024-12-06T22:53:12 (ISO-8601, seconds precision, no TZ)
        row["created_at"] = row["created_at"].isoformat(timespec="seconds")
        f.write(json.dumps(row, ensure_ascii=False) + "\n")   # newline-delimited JSON (jsonl)

print(f"Wrote {len(merged_ds):,} lines to {out_path.resolve()}")