from __future__ import annotations
import argparse
import json
import random
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict

def month_key(iso_ts: str) -> str:
    try:
        # Drop the trailing 'Z' (if any) so fromisoformat works.
        ts_clean = iso_ts.rstrip("Z")
        dt = datetime.fromisoformat(ts_clean)
        return f"{dt.year:04d}-{dt.month:02d}"
    except (ValueError, TypeError):
        return iso_ts[:7]

def load_by_month(path: Path, start_bound, end_bound) -> dict[str, list[dict]]:
    """Group JSON-line objects by YYYY-MM, filtering by date bounds (inclusive)."""
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

            ts_raw = obj.get("created_at", "")
            try:
                ts_clean = ts_raw.rstrip("Z")
                dt = datetime.fromisoformat(ts_clean)
                dt_month = date(dt.year, dt.month, 1)
            except (ValueError, TypeError):
                print(f"⤤ bad timestamp at line {line_no}: {ts_raw}")
                continue

            # Filter based on the month boundaries
            if start_bound and dt_month < start_bound:
                continue
            if end_bound and dt_month > end_bound:
                continue

            key = f"{dt.year:04d}-{dt.month:02d}"
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

def parse_month(s: str) -> date:
    """Return a date for the first day of the given YYYY-MM."""
    return datetime.strptime(s, "%Y-%m").date().replace(day=1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Create lite dataset by sampling from full dataset")
    parser.add_argument(
        "--input-file", 
        type=Path,
        help="Input full dataset file (default: datasets/full-{today}.jsonl)"
    )
    parser.add_argument(
        "--output-file", 
        type=Path,
        help="Output lite dataset file (default: datasets/lite-{today}.jsonl)"
    )
    parser.add_argument(
        "--samples-per-month", 
        type=int, 
        default=50,
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
    )
    parser.add_argument(
    "--start-month",
    type=str,
    help="Earliest month to **include** (format YYYY-MM, e.g. 2024-12)",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        help="Latest month to **include** (format YYYY-MM, e.g. 2025-05)",
    )
    
    args = parser.parse_args()

    try:
        start_bound = parse_month(args.start_month) if args.start_month else None
        end_bound   = parse_month(args.end_month)   if args.end_month   else False
    except ValueError as e:
        raise SystemExit(f"month argument must be YYYY-MM: {e}")
    
    # Set default file paths if not provided
    today = date.today().isoformat()
    if args.input_file is None:
        args.input_file = Path("datasets") / f"full-{today}.jsonl"
    if args.output_file is None:
        args.output_file = Path("datasets") / f"lite-{today}.jsonl"
    
    # Set random seed
    random.seed(args.random_seed)
    
    if not args.input_file.exists():
        raise SystemExit(f"Input file {args.input_file} does not exist")
    
    groups  = load_by_month(args.input_file, start_bound, end_bound)
    subset  = sample_groups(groups, k=args.samples_per_month)

    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with args.output_file.open("w", encoding="utf-8") as out:
        for obj in subset:
            json.dump(obj, out, ensure_ascii=False)
            out.write("\n")

    print(f"Subset ({len(subset)} instances) written to {args.output_file.resolve()}")

if __name__ == "__main__":
    main()

