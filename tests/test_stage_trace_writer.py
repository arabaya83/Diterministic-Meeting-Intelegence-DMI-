from __future__ import annotations

import json
from pathlib import Path

from ami_mom_pipeline.utils.traceability import StageTraceWriter


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_stage_trace_writer_truncates_on_new_run(tmp_path: Path) -> None:
    trace_path = tmp_path / "artifacts" / "ami" / "ES2005a" / "stage_trace.jsonl"

    writer1 = StageTraceWriter(trace_path, enabled=True, truncate_on_init=True)
    writer1.write({"event": "stage_start", "run": 1, "stage": "ingest"})
    writer1.write({"event": "stage_end", "run": 1, "stage": "ingest", "status": "ok"})
    rows1 = _read_jsonl(trace_path)
    assert len(rows1) == 2
    assert all(r["run"] == 1 for r in rows1)

    writer2 = StageTraceWriter(trace_path, enabled=True, truncate_on_init=True)
    writer2.write({"event": "stage_start", "run": 2, "stage": "ingest"})
    rows2 = _read_jsonl(trace_path)
    assert len(rows2) == 1
    assert rows2[0]["run"] == 2
    assert rows2[0]["event"] == "stage_start"
