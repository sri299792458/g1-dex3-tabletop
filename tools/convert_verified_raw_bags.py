#!/usr/bin/env python3
"""Convert raw G1 MCAP runs sequentially and remove only verified source bags."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIRMATION = "I CONFIRM VERIFIED RAW BAGS MAY BE PERMANENTLY DELETED"
STATUSES = ("completed", "task_rejected", "failed")
REQUIRED_TOPICS = (
    "/lowstate",
    "/lowcmd",
    "/secondary_imu",
    "/dex3/left/state",
    "/dex3/right/state",
    "/dex3/left/cmd",
    "/dex3/right/cmd",
    "/camera/color/image_raw",
    "/camera/depth/image_rect_raw",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_descriptor(run: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    status = read_json(run / "status.json")
    manifest = read_json(run / "raw_episode" / "episode_manifest.json")
    episode_id = str((manifest.get("episode") or {}).get("episode_id") or run.name)
    return status, manifest, episode_id


def missing_required_topics(manifest: dict[str, Any]) -> list[str]:
    topics = (((manifest.get("capture") or {}).get("bag") or {}).get("topics") or {})
    return [
        topic
        for topic in REQUIRED_TOPICS
        if int((topics.get(topic) or {}).get("message_count", 0) or 0) <= 0
    ]


def dataset_id(prefix: str, status: str) -> str:
    return f"{prefix}_{status}"


def verified_summaries(dataset_root: Path) -> list[Path]:
    summaries = sorted(dataset_root.glob("meta/g1_conversion/*/conversion_summary.json"))
    return [path for path in summaries if read_json(path).get("status") == "verified"]


def assert_dataset_consistent(dataset_root: Path) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        if dataset_root.exists() and any(dataset_root.iterdir()):
            raise RuntimeError(f"dataset target is nonempty but uninitialized: {dataset_root}")
        return
    total_episodes = int(read_json(info_path).get("total_episodes", -1))
    verified = len(verified_summaries(dataset_root))
    if total_episodes != verified:
        raise RuntimeError(
            f"dataset has {total_episodes} episodes but {verified} verified conversion summaries: "
            f"{dataset_root}"
        )


def conversion_summary(dataset_root: Path, episode_id: str) -> dict[str, Any] | None:
    path = dataset_root / "meta" / "g1_conversion" / episode_id / "conversion_summary.json"
    if not path.is_file():
        return None
    value = read_json(path)
    return value if value.get("status") == "verified" else None


def safe_remove_bag(run: Path, runs_root: Path) -> list[dict[str, Any]]:
    bag = run / "raw_episode" / "bag"
    if bag.is_symlink() or not bag.is_dir():
        raise RuntimeError(f"raw bag is not a normal directory: {bag}")
    resolved_root = runs_root.resolve()
    resolved_run = run.resolve()
    resolved_bag = bag.resolve()
    if resolved_run.parent != resolved_root or resolved_bag != resolved_run / "raw_episode" / "bag":
        raise RuntimeError(f"refusing unexpected raw bag path: {resolved_bag}")

    entries: list[dict[str, Any]] = []
    for path in sorted(bag.rglob("*")):
        stat = path.lstat()
        if path.is_symlink():
            kind = "symlink"
        elif path.is_file():
            kind = "file"
        elif path.is_dir():
            kind = "directory"
        else:
            raise RuntimeError(f"refusing special file in raw bag: {path}")
        entries.append(
            {
                "relative_path": str(path.relative_to(bag)),
                "size_bytes": stat.st_size if path.is_file() else 0,
                "kind": kind,
            }
        )
    if not any(entry["relative_path"].endswith(".mcap") for entry in entries):
        raise RuntimeError(f"raw bag contains no MCAP file: {bag}")

    for path in sorted(bag.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    bag.rmdir()
    return entries


def build_parser() -> argparse.ArgumentParser:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=workspace / "runs")
    parser.add_argument("--published-root", type=Path, default=workspace / "published")
    parser.add_argument("--dataset-prefix", default="rpm_lab/g1_dex3_tabletop")
    parser.add_argument("--status", action="append", choices=STATUSES, dest="statuses")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and args.confirm != CONFIRMATION:
        raise RuntimeError(f"destructive conversion requires --confirm {CONFIRMATION!r}")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    workspace = Path(__file__).resolve().parents[1]
    runs_root = args.runs_root.expanduser().resolve()
    published_root = args.published_root.expanduser().resolve()
    statuses = tuple(args.statuses or STATUSES)
    work = workspace / "work" / "lerobot_bulk_conversion"
    ledger = work / "ledger.jsonl"
    lock_path = work / "batch.lock"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    for run in sorted(runs_root.iterdir()):
        bag = run / "raw_episode" / "bag"
        if run.is_dir() and bag.is_dir() and any(bag.glob("*.mcap")):
            candidates.append(run)
    if args.max_runs is not None:
        candidates = candidates[: args.max_runs]

    selected = 0
    converted = 0
    deleted_bytes = 0
    failed = 0
    skipped_incompatible = 0
    blocked_statuses: set[str] = set()

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another batch conversion owns {lock_path}") from error

        for run in candidates:
            try:
                status_doc, manifest, episode_id = source_descriptor(run)
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                failed += 1
                append_jsonl(
                    ledger,
                    {"at": utc_now(), "run": run.name, "result": "metadata_error", "error": str(error)},
                )
                print(f"SKIP {run.name}: metadata error: {error}", flush=True)
                continue
            status = str(status_doc.get("status", ""))
            if status not in statuses:
                continue
            selected += 1
            missing = missing_required_topics(manifest)
            if missing:
                skipped_incompatible += 1
                append_jsonl(
                    ledger,
                    {
                        "at": utc_now(),
                        "run": run.name,
                        "status": status,
                        "result": "retained_incompatible",
                        "missing_topics": missing,
                    },
                )
                print(f"RETAIN {run.name}: missing {missing}", flush=True)
                continue
            if status in blocked_statuses:
                print(f"RETAIN {run.name}: dataset for status {status} is blocked", flush=True)
                continue

            repo_id = dataset_id(args.dataset_prefix, status)
            dataset_root = published_root / repo_id
            try:
                assert_dataset_consistent(dataset_root)
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                blocked_statuses.add(status)
                failed += 1
                append_jsonl(
                    ledger,
                    {
                        "at": utc_now(),
                        "run": run.name,
                        "status": status,
                        "dataset_id": repo_id,
                        "result": "dataset_blocked",
                        "error": str(error),
                    },
                )
                print(f"BLOCK {status}: {error}", flush=True)
                continue

            summary = conversion_summary(dataset_root, episode_id)
            if args.dry_run:
                action = "delete_verified" if summary else "convert_then_delete"
                print(f"DRY-RUN {run.name}: {action} -> {repo_id}", flush=True)
                continue

            if summary is None:
                log_path = logs / f"{run.name}.log"
                command = [
                    str(workspace / "tools" / "convert_raw_to_lerobot.sh"),
                    str(run),
                    "--dataset-id",
                    repo_id,
                    "--published-root",
                    str(published_root),
                ]
                if status != "completed":
                    command.append("--allow-non-completed")
                print(f"CONVERT {run.name} -> {repo_id}", flush=True)
                with log_path.open("w", encoding="utf-8") as output:
                    process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
                if process.returncode != 0:
                    failed += 1
                    try:
                        assert_dataset_consistent(dataset_root)
                    except (OSError, ValueError, TypeError, RuntimeError) as error:
                        blocked_statuses.add(status)
                        detail = f"; dataset blocked: {error}"
                    else:
                        detail = ""
                    append_jsonl(
                        ledger,
                        {
                            "at": utc_now(),
                            "run": run.name,
                            "status": status,
                            "dataset_id": repo_id,
                            "result": "conversion_failed",
                            "exit_code": process.returncode,
                            "log": str(log_path),
                            "detail": detail,
                        },
                    )
                    print(f"RETAIN {run.name}: conversion failed{detail}; log={log_path}", flush=True)
                    continue
                summary = conversion_summary(dataset_root, episode_id)
                if summary is None:
                    raise RuntimeError(f"converter exited successfully without a verified summary for {run.name}")
                assert_dataset_consistent(dataset_root)
                converted += 1

            original_files = [
                {
                    "relative_path": str(path.relative_to(run / "raw_episode" / "bag")),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted((run / "raw_episode" / "bag").rglob("*"))
                if path.is_file()
            ]
            original_bytes = sum(item["size_bytes"] for item in original_files)
            summary_path = (
                dataset_root / "meta" / "g1_conversion" / episode_id / "conversion_summary.json"
            )
            receipt = {
                "schema_version": 1,
                "episode_id": episode_id,
                "source_run": str(run),
                "source_status": status,
                "dataset_id": repo_id,
                "dataset_root": str(dataset_root),
                "dataset_episode_index": summary["dataset_episode_index"],
                "published_frame_count": summary["published_frame_count"],
                "conversion_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "original_bag_files": original_files,
                "original_size_bytes": original_bytes,
                "verified_at": utc_now(),
                "deletion_policy": "permanent_after_verified_lerobot_reload",
            }
            receipt_path = run / "raw_episode" / "lerobot_replacement.json"
            write_json_atomic(receipt_path, receipt)
            safe_remove_bag(run, runs_root)
            deleted_bytes += original_bytes
            append_jsonl(
                ledger,
                {
                    "at": utc_now(),
                    "run": run.name,
                    "status": status,
                    "dataset_id": repo_id,
                    "dataset_episode_index": summary["dataset_episode_index"],
                    "result": "converted_verified_raw_deleted",
                    "deleted_bytes": original_bytes,
                    "receipt": str(receipt_path),
                },
            )
            print(
                f"DELETED {run.name}/raw_episode/bag after verification; freed={original_bytes / 2**30:.3f} GiB",
                flush=True,
            )

    report = {
        "finished_at": utc_now(),
        "selected": selected,
        "converted_this_run": converted,
        "deleted_gib": deleted_bytes / 2**30,
        "conversion_failures": failed,
        "retained_incompatible": skipped_incompatible,
        "blocked_statuses": sorted(blocked_statuses),
        "ledger": str(ledger),
    }
    write_json_atomic(work / "latest_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if failed or blocked_statuses else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
