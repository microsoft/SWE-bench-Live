from __future__ import annotations
import json
import os
from datetime import date
from pathlib import Path

# -------- configurable paths --------
ROOT = Path("logs/run_evaluation/tutorial-validation/gold")
OUTPUT_DIR = Path("datasets")
TODAY = date.today().isoformat()          # e.g. "2025-06-07"
OUT_PATH = OUTPUT_DIR / f"full-{TODAY}.jsonl"
# ------------------------------------

def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"Expected directory {ROOT} not found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Overwrite existing file for today; change to "a" to append instead.
    with OUT_PATH.open("w", encoding="utf-8") as outfile:
        for instance_path in ROOT.rglob("instance.json"):
            try:
                with instance_path.open(encoding="utf-8") as fp:
                    dct = json.load(fp)
            except (json.JSONDecodeError, OSError) as err:
                # Corrupt or unreadable JSON—skip it.
                print(f"Skipping {instance_path}: {err}")
                continue

            if dct.get("FAIL_TO_PASS") and dct.get("PASS_TO_PASS"):
                json.dump(dct, outfile, ensure_ascii=False)
                outfile.write("\n")

    print(f"Collected records written to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()

