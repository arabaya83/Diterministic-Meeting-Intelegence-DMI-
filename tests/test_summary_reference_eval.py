from __future__ import annotations

import csv
import json
from pathlib import Path

from ami_mom_pipeline.config import AppConfig
from ami_mom_pipeline.pipeline import PipelinePaths, stage_evaluate, stage_finalize_summary
from ami_mom_pipeline.utils.ami_annotations import load_abstractive_summary_sections, load_abstractive_summary_text


def test_load_abstractive_summary_sections_reads_reference_xml(tmp_path: Path) -> None:
    annotations_dir = tmp_path / "annotations"
    abstractive_dir = annotations_dir / "abstractive"
    abstractive_dir.mkdir(parents=True)
    (abstractive_dir / "TEST100a.abssumm.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <abstract nite:id="TEST100a.abstract.1">
    <sentence nite:id="TEST100a.s.1">The team reviewed the concept design.</sentence>
    <sentence nite:id="TEST100a.s.2">They compared cost and usability concerns.</sentence>
  </abstract>
  <actions nite:id="TEST100a.actions.1">
    <sentence nite:id="TEST100a.s.3">The designer will prepare a prototype.</sentence>
  </actions>
</nite:root>
""",
        encoding="utf-8",
    )

    sections = load_abstractive_summary_sections(annotations_dir, "TEST100a")

    assert sections["abstract"] == [
        "The team reviewed the concept design.",
        "They compared cost and usability concerns.",
    ]
    assert sections["actions"] == ["The designer will prepare a prototype."]
    assert load_abstractive_summary_text(annotations_dir, "TEST100a", section="abstract") == (
        "The team reviewed the concept design. They compared cost and usability concerns."
    )


def test_stage_evaluate_writes_real_rouge_scores_when_reference_exists(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.paths.annotations_dir = str(tmp_path / "annotations")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    annotations_dir = Path(cfg.paths.annotations_dir)
    eval_dir = Path(cfg.paths.artifacts_dir) / "eval" / "ami"
    eval_dir.mkdir(parents=True)
    (annotations_dir / "abstractive").mkdir(parents=True)
    (annotations_dir / "abstractive" / "TEST100a.abssumm.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <abstract nite:id="TEST100a.abstract.1">
    <sentence nite:id="TEST100a.s.1">The team reviewed the concept design and compared costs.</sentence>
    <sentence nite:id="TEST100a.s.2">They planned to prepare a prototype.</sentence>
  </abstract>
</nite:root>
""",
        encoding="utf-8",
    )
    (annotations_dir / "words").mkdir(parents=True)
    (annotations_dir / "words" / "TEST100a.A.words.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w starttime="0.00" endtime="1.00">hello</w>
  <w starttime="1.00" endtime="2.00">world</w>
</nite:root>
""",
        encoding="utf-8",
    )

    paths = PipelinePaths(
        raw_audio_dir=tmp_path / "audio",
        annotations_dir=annotations_dir,
        staged_dir=tmp_path / "staged",
        artifacts_dir=Path(cfg.paths.artifacts_dir),
        meeting_artifacts_dir=Path(cfg.paths.artifacts_dir) / "ami" / "TEST100a",
        eval_dir=eval_dir,
    )
    paths.meeting_artifacts_dir.mkdir(parents=True)
    (paths.meeting_artifacts_dir / "asr_segments.json").write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "SPEAKER_1",
                    "text": "hello world",
                    "source": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    (paths.meeting_artifacts_dir / "diarization_segments.json").write_text(
        json.dumps(
            [
                {
                    "start": 0.25,
                    "end": 1.75,
                    "speaker": "SPEAKER_1",
                    "source": "test",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = stage_evaluate(
        cfg=cfg,
        paths=paths,
        meeting_id="TEST100a",
        asr_out={"segments": [{"text": "hello world"}]},
        token_cache=[{"text": "hello", "is_punc": False}, {"text": "world", "is_punc": False}],
        summary_out={
            "summary": "The team reviewed the concept design and compared costs. They planned to prepare a prototype.",
            "key_points": ["concept design"],
        },
        extract_out={"decisions": [], "action_items": []},
    )

    assert result["rouge1"] is not None
    assert result["rouge2"] is not None
    assert result["rougeL"] is not None
    assert result["rouge1"] > 0.9
    assert result["cpwer"] == 0.0
    assert result["der"] == 0.0

    with (eval_dir / "rouge_scores.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["meeting_id"] == "TEST100a"
    assert rows[0]["rouge1"] != ""
    assert rows[0]["rouge2"] != ""
    assert rows[0]["rougeL"] != ""

    with (eval_dir / "wer_scores.csv").open("r", encoding="utf-8", newline="") as handle:
        wer_rows = list(csv.DictReader(handle))
    assert wer_rows[0]["cpwer"] == "0.000000"
    assert wer_rows[0]["der"] == "0.000000"

    with (eval_dir / "speech_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        speech_rows = list(csv.DictReader(handle))
    assert speech_rows[0]["cpwer"] == "0.000000"
    assert speech_rows[0]["der"] == "0.000000"

    wer_breakdown = json.loads((eval_dir / "wer_breakdown.json").read_text(encoding="utf-8"))
    assert wer_breakdown["meeting_id"] == "TEST100a"
    assert wer_breakdown["cpwer"] == 0.0
    assert wer_breakdown["der"] == 0.0


def test_stage_finalize_summary_enriches_narrative_with_missing_action_or_decision(tmp_path: Path) -> None:
    cfg = AppConfig()
    paths = tmp_path / "meeting"
    paths.mkdir()
    summary_out = {
        "meeting_id": "TEST100a",
        "summary": "The team discussed concept design options and cost constraints.",
        "key_points": ["Concept design options"],
        "discussion_points": [],
        "follow_up": [],
        "prompt_template_version": "llama-cpp-v1",
        "backend": "llama_cpp",
    }
    extract_out = {
        "meeting_id": "TEST100a",
        "decisions": [
            {
                "decision": "using a wheel interface",
                "evidence_chunk_ids": ["c1"],
                "evidence_snippets": ["They chose a wheel interface."],
                "confidence": 0.7,
                "uncertain": False,
            }
        ],
        "action_items": [
            {
                "action": "Prepare a prototype for the next meeting",
                "owner": None,
                "due_date": None,
                "evidence_chunk_ids": ["c2"],
                "evidence_snippets": ["Prepare a prototype for the next meeting."],
                "confidence": 0.72,
                "uncertain": False,
            }
        ],
        "flags": [],
    }

    class _Paths:
        meeting_artifacts_dir = paths

    finalized = stage_finalize_summary(cfg, _Paths(), "TEST100a", summary_out, extract_out)
    assert "using a wheel interface" in finalized["summary"].lower()
    assert "prepare a prototype" in finalized["summary"].lower()
