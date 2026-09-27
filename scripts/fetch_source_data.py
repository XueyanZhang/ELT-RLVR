"""Fetch the source tables for chosen tasks, without downloading the archive.

The benchmark ships every task's source data in one 4 GB zip, dominated by a
handful of very large tasks. Training uses small ones, so their tables are
pulled individually over HTTP range requests - the tasks in the default pool
come to a few megabytes rather than four gigabytes.

    python scripts/fetch_source_data.py --tasks shipping,github
    python scripts/fetch_source_data.py --list
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
# The benchmark splits its source data across two archives: the REST-API
# tables live in one, everything else in the other. Neither is complete on its
# own, so a task is assembled from both.
ARCHIVE_ID = "11vQqNEWXoPG6sjKytAn7TtFLDMiQa17I"   # postgres, mongodb, s3, flat files
API_DATA = REPO / "data/raw/api_data"              # the REST-API tables
DEST = REPO / "data/source"


def archive_url() -> str:
    """Resolve the archive's download URL past Google Drive's confirm page."""
    page = subprocess.run(
        ["curl", "-sL", "--max-time", "60",
         f"https://drive.google.com/uc?export=download&id={ARCHIVE_ID}"],
        capture_output=True, text=True, check=True,
    ).stdout
    match = re.search(r'name="uuid" value="([^"]*)"', page)
    if not match:
        raise RuntimeError("Drive did not return a confirm token; the link may have moved.")
    return (
        f"https://drive.usercontent.google.com/download?id={ARCHIVE_ID}"
        f"&export=download&confirm=t&uuid={match.group(1)}"
    )


def _fetch_flat_files(task: str, out: Path) -> None:
    """Pull the tables the benchmark serves as downloadable files.

    A handful of tables are not in the archive at all: the task declares them
    as `flat_files` pointing at a URL, which is one of the five source kinds
    the benchmark deliberately includes. Fetched once and kept locally, so a
    rollout never depends on reaching them.
    """
    config_path = REPO / f"third_party/ELT-Bench/elt-bench/snowflake/{task}/config.yaml"
    if not config_path.exists():
        return
    for entry in yaml.safe_load(config_path.read_text()).get("flat_files") or []:
        target = out / f"{entry['table']}.csv"
        if target.exists() or not entry.get("path"):
            continue
        body = subprocess.run(
            ["curl", "-sL", "--max-time", "120", entry["path"]],
            capture_output=True, check=False,
        ).stdout
        if not body or b"<html" in body[:400].lower():
            continue
        if entry.get("format") == "jsonl":
            import io
            import pandas as pd
            frame = pd.read_json(io.BytesIO(body), lines=True, dtype=str)
            frame.to_csv(target, index=False)
        else:
            target.write_bytes(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="", help="comma-separated task names")
    parser.add_argument("--max-rows", type=int, default=10_000,
                        help="with no --tasks, take every task under this row count")
    parser.add_argument("--list", action="store_true", help="only report what would be fetched")
    args = parser.parse_args()

    expected = json.loads((REPO / "third_party/ELT-Bench/evaluation/table.json").read_text())
    if args.tasks:
        tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    else:
        tasks = sorted(t for t in expected if sum(expected[t].values()) <= args.max_rows)

    unknown = [t for t in tasks if t not in expected]
    if unknown:
        print(f"unknown tasks: {', '.join(unknown)}", file=sys.stderr)
        return 1

    from remotezip import RemoteZip

    # Drive throttles bursts of range requests with a 503, so the connection is
    # retried rather than treated as a failure.
    archive = None
    for attempt in range(6):
        try:
            archive = RemoteZip(archive_url())
            break
        except Exception as error:
            wait = 5 * 2**attempt
            print(f"  archive unavailable ({error}); retrying in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    if archive is None:
        print("could not reach the archive after several attempts", file=sys.stderr)
        return 1

    with archive:
        members = [n for n in archive.namelist() if n.endswith(".csv")]
        by_task: dict[str, list[str]] = {}
        for name in members:
            parts = name.split("/")
            if len(parts) == 3 and parts[1] in set(tasks):
                by_task.setdefault(parts[1], []).append(name)

        total_bytes = sum(
            archive.getinfo(n).file_size for names in by_task.values() for n in names
        )
        print(f"{len(tasks)} tasks, {sum(len(v) for v in by_task.values())} tables, "
              f"{total_bytes / 1e6:.1f} MB")
        if args.list:
            for task in sorted(tasks):
                from_api = {p.stem for p in (API_DATA / task).glob("*.csv")} if (API_DATA / task).is_dir() else set()
                missing = set(expected[task]) - {Path(n).stem for n in by_task.get(task, [])} - from_api
                flag = f"  MISSING {sorted(missing)}" if missing else ""
                print(f"  {task:<28}{len(expected[task]):>3} tables{flag}")
            return 0

        for task in sorted(tasks):
            out = DEST / task
            out.mkdir(parents=True, exist_ok=True)
            for name in by_task.get(task, []):
                target = out / Path(name).name
                if target.exists():
                    continue
                for attempt in range(5):
                    try:
                        target.write_bytes(archive.read(name))
                        break
                    except Exception:
                        time.sleep(3 * 2**attempt)
            api_source = API_DATA / task
            if api_source.is_dir():
                for csv in api_source.glob("*.csv"):
                    target = out / csv.name
                    if not target.exists():
                        target.write_bytes(csv.read_bytes())
            _fetch_flat_files(task, out)
            found = {p.stem.lower() for p in out.glob("*.csv")}
            missing = {t for t in expected[task] if t.lower() not in found}
            status = f"MISSING {sorted(missing)}" if missing else "ok"
            print(f"  {task:<28}{len(found):>3} tables  {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
