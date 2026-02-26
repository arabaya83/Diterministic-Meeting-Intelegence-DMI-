from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(stable_json_dumps(data), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            f.write("\n")


def upsert_jsonl(path: Path, row: dict[str, Any], key: str) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing.append(json.loads(line))
    replaced = False
    for i, rec in enumerate(existing):
        if rec.get(key) == row.get(key):
            existing[i] = row
            replaced = True
            break
    if not replaced:
        existing.append(row)
    existing.sort(key=lambda r: str(r.get(key, "")))
    write_jsonl(path, existing)


def upsert_csv(path: Path, row: dict[str, Any], key: str) -> None:
    ensure_dir(path.parent)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    replaced = False
    for i, rec in enumerate(rows):
        if rec.get(key) == str(row.get(key)):
            rows[i] = {k: str(v) for k, v in row.items()}
            replaced = True
            break
    if not replaced:
        rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: r.get(key, ""))
    fieldnames = list(rows[0].keys()) if rows else list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

