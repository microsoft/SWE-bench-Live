from __future__ import annotations
import argparse
from collections import defaultdict
import datetime
import json
import random
from pathlib import Path
from datetime import date, datetime
from swebench.collect.produce.utilities.verification import Verifier


def parse_month(s: str) -> date:
    """Return a date for the first day of the given YYYY-MM."""
    return datetime.strptime(s, "%Y-%m").date().replace(day=1)

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

    result = []
    for month in groups.keys():
        result.extend(groups[month])
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Create verified dataset by prompting reasoing model")
    parser.add_argument(
        "--input-file", 
        type=Path,
        help="Input full dataset file (default: datasets/full-{today}.jsonl)"
    )
    parser.add_argument(
        "--output-file", 
        type=Path,
        help="Output dataset file (default: datasets/verified-{today}.jsonl)"
    )
    parser.add_argument(
        "--log-file", 
        type=Path,
        help="Output dataset file (default: datasets/verified-log-{today}.jsonl)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["OpenAI", "AOAI", "Anthropic"],
        help="The LLM provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model version, such as o3-20250416",
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
        args.output_file = Path("datasets") / f"verified-{today}.jsonl"
    if args.log_file is None:
        args.log_file = Path("datasets") / f"verified-log-{today}.jsonl"
    
    if not args.input_file.exists():
        raise SystemExit(f"Input file {args.input_file} does not exist")
    samples = load_by_month(args.input_file, start_bound, end_bound)

    verifier = Verifier(args.provider, args.model)
    records = verifier.analyse_all(samples)
    retain_id = [record["instance_id"] for record in records if record["category"] == "7"]
    subset = [sample for sample in samples if sample["instance_id"] in retain_id]

    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with args.output_file.open("w", encoding="utf-8") as out:
        for obj in subset:
            json.dump(obj, out, ensure_ascii=False)
            out.write("\n")
    print(f"Subset ({len(subset)} instances) written to {args.output_file.resolve()}")

    with args.log_file.open("w", encoding="utf-8") as out:
        for obj in records:
            json.dump(obj, out, ensure_ascii=False)
            out.write("\n")

    print(f"LLM decision ({len(records)} instances) written to {args.log_file.resolve()}")

if __name__ == "__main__":
    main()

