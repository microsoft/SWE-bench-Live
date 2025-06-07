from __future__ import annotations
import json
import random
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict

# ------------- configurable paths -------------
TODAY = date.today().isoformat()                   # "2025-06-07"
DATA_DIR = Path("datasets")
IN_PATH  = DATA_DIR / f"full-{TODAY}.jsonl"
OUT_PATH = DATA_DIR / f"lite-{TODAY}.jsonl"
# ----------------------------------------------

def month_key(iso_ts: str) -> str:
    """
    Convert '2025-06-01T13:40:01Z' --> '2025-06'
    Assumes Zulu or offset; we just parse the date part.
    """
    try:
        # Drop the trailing 'Z' (if any) so fromisoformat works.
        ts_clean = iso_ts.rstrip("Z")
        dt = datetime.fromisoformat(ts_clean)
        return f"{dt.year:04d}-{dt.month:02d}"
    except (ValueError, TypeError):
        # Fallback: slice the first 7 chars if weird formatting.
        return iso_ts[:7]

def load_by_month(path: Path) -> dict[str, list[dict]]:
    """Group JSON-line objects by YYYY-MM."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⤤ bad JSON at line {line_no}: {e}")
                continue

            key = month_key(obj.get("created_at", ""))
            groups[key].append(obj)
    return groups

def sample_groups(groups: dict[str, list[dict]], k: int = 50) -> list[dict]:
    """Randomly sample up to *k* items from each month's list."""
    sampled: list[dict] = []
    for month, items in groups.items():
        if len(items) > k:
            sampled.extend(random.sample(items, k))
        else:
            sampled.extend(items)
    return sampled

def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Input file {IN_PATH} does not exist")
    
    groups  = load_by_month(IN_PATH)
    subset  = sample_groups(groups, k=50)

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for obj in subset:
            json.dump(obj, out, ensure_ascii=False)
            out.write("\n")

    print(f"Subset ({len(subset)} instances) written to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    random.seed(42)
    main()

