"""Speech-evaluation helpers for persisted AMI artifacts.

This module reads reference AMI annotations and pipeline-generated speech
artifacts, then computes lightweight deterministic metrics such as WER, cpWER,
and an approximate no-overlap DER. The logic is intentionally local and
dependency-light so evaluation can run offline.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ami_annotations import build_utterances, load_word_tokens, reference_plain_text


@dataclass
class WordErrorCounts:
    """Edit-count summary used to derive WER-like metrics."""

    edits: int
    ref_len: int


def normalize_for_eval(text: str) -> str:
    """Normalize transcript text into a token space suitable for scoring."""
    import re

    s = text.lower()
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"speaker_\d+:", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def word_error_counts(ref_text: str, hyp_text: str) -> WordErrorCounts:
    """Return raw word-level edit counts for two transcript strings."""
    ref_words = normalize_for_eval(ref_text).split()
    hyp_words = normalize_for_eval(hyp_text).split()
    if not ref_words:
        return WordErrorCounts(edits=(0 if not hyp_words else len(hyp_words)), ref_len=0)
    return WordErrorCounts(edits=edit_distance(ref_words, hyp_words), ref_len=len(ref_words))


def wer_from_texts(ref_text: str, hyp_text: str) -> float | None:
    """Compute WER for two transcript strings when a reference exists."""
    counts = word_error_counts(ref_text, hyp_text)
    if counts.ref_len == 0:
        return None
    return counts.edits / counts.ref_len


def load_reference_asr_views(annotations_dir: Path, meeting_id: str) -> dict[str, Any]:
    """Load AMI reference tokens and aggregate them into speaker/text views."""
    tokens = load_word_tokens(annotations_dir, meeting_id)
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for t in tokens:
        by_speaker.setdefault(str(t["speaker_letter"]), []).append(t)
    speaker_texts = {
        spk: reference_plain_text(spk_tokens)
        for spk, spk_tokens in sorted(by_speaker.items(), key=lambda kv: kv[0])
    }
    return {
        "tokens": tokens,
        "full_text": reference_plain_text(tokens),
        "speaker_texts": speaker_texts,
        "speaker_count": len(speaker_texts),
    }


def load_hyp_asr_views(artifacts_dir: Path, meeting_id: str) -> dict[str, Any]:
    """Load pipeline ASR artifacts and aggregate them for scoring."""
    asr_path = artifacts_dir / "ami" / meeting_id / "asr_segments.json"
    if not asr_path.exists():
        return {"exists": False, "segments": [], "full_text": "", "speaker_texts": {}, "speaker_count": 0}
    data = json.loads(asr_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{asr_path} must contain a list")
    segments: list[dict[str, Any]] = []
    by_speaker_parts: dict[str, list[str]] = {}
    all_parts: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        segments.append(item)
        speaker = str(item.get("speaker", "UNKNOWN"))
        text = str(item.get("text", "")).strip()
        if text:
            by_speaker_parts.setdefault(speaker, []).append(text)
            all_parts.append(text)
    speaker_texts = {k: " ".join(v).strip() for k, v in sorted(by_speaker_parts.items(), key=lambda kv: kv[0])}
    return {
        "exists": True,
        "segments": segments,
        "full_text": " ".join(all_parts).strip(),
        "speaker_texts": speaker_texts,
        "speaker_count": len(speaker_texts),
    }


def compute_cpwer(ref_speaker_texts: dict[str, str], hyp_speaker_texts: dict[str, str]) -> dict[str, Any]:
    """Compute cpWER via brute-force speaker assignment for small meetings."""
    ref_items = sorted(ref_speaker_texts.items(), key=lambda kv: kv[0])
    hyp_items = sorted(hyp_speaker_texts.items(), key=lambda kv: kv[0])
    ref_labels = [k for k, _ in ref_items]
    hyp_labels = [k for k, _ in hyp_items]
    ref_texts = [v for _, v in ref_items]
    hyp_texts = [v for _, v in hyp_items]

    total_ref_words = len(normalize_for_eval(" ".join(ref_texts)).split())
    if total_ref_words == 0:
        return {
            "cpwer": None,
            "best_assignment": [],
            "total_ref_words": 0,
            "total_edits": None,
        }

    n = max(len(ref_texts), len(hyp_texts))
    if n == 0:
        return {"cpwer": None, "best_assignment": [], "total_ref_words": total_ref_words, "total_edits": None}
    if n > 8:
        raise ValueError(f"cpWER brute-force assignment capped at 8 speakers, got {n}")

    ref_texts_pad = ref_texts + [""] * (n - len(ref_texts))
    hyp_texts_pad = hyp_texts + [""] * (n - len(hyp_texts))
    ref_labels_pad = ref_labels + [f"__ref_dummy_{i}" for i in range(n - len(ref_labels))]
    hyp_labels_pad = hyp_labels + [f"__hyp_dummy_{i}" for i in range(n - len(hyp_labels))]

    cost: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        for j in range(n):
            row.append(word_error_counts(ref_texts_pad[i], hyp_texts_pad[j]).edits)
        cost.append(row)

    best_cost: int | None = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(n)):
        c = 0
        for i, j in enumerate(perm):
            c += cost[i][j]
            if best_cost is not None and c >= best_cost:
                break
        if best_cost is None or c < best_cost:
            best_cost = c
            best_perm = perm

    assert best_perm is not None and best_cost is not None
    assignment = []
    for i, j in enumerate(best_perm):
        if not ref_labels_pad[i].startswith("__ref_dummy_") and not hyp_labels_pad[j].startswith("__hyp_dummy_"):
            assignment.append({"ref_speaker": ref_labels_pad[i], "hyp_speaker": hyp_labels_pad[j]})

    return {
        "cpwer": best_cost / total_ref_words,
        "best_assignment": assignment,
        "total_ref_words": total_ref_words,
        "total_edits": best_cost,
    }


def load_ref_diarization_from_words(annotations_dir: Path, meeting_id: str) -> list[dict[str, Any]]:
    """Build reference diarization segments from AMI word annotations."""
    tokens = load_word_tokens(annotations_dir, meeting_id)
    utts = build_utterances(tokens)
    return [
        {
            "start": float(u["start"]),
            "end": float(u["end"]),
            "speaker": str(u["speaker_letter"]),
            "source": "ami_words_utterance_grouping",
        }
        for u in utts
        if float(u["end"]) > float(u["start"])
    ]


def load_hyp_diarization(artifacts_dir: Path, meeting_id: str) -> list[dict[str, Any]]:
    """Load diarization segments emitted by the pipeline for one meeting."""
    diar_path = artifacts_dir / "ami" / meeting_id / "diarization_segments.json"
    if not diar_path.exists():
        return []
    data = json.loads(diar_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{diar_path} must contain a list")
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        out.append({"start": start, "end": end, "speaker": str(item.get("speaker", "UNKNOWN"))})
    return out


def compute_der_approx_nooverlap(
    ref_segments: list[dict[str, Any]],
    hyp_segments: list[dict[str, Any]],
    collar_sec: float = 0.25,
    skip_overlap: bool = True,
) -> dict[str, Any]:
    """Compute a deterministic approximate DER over single-label intervals."""
    if collar_sec < 0:
        raise ValueError("collar_sec must be >= 0")

    ref_proc = _apply_collar(ref_segments, collar_sec)
    hyp_proc = [s for s in hyp_segments if s["end"] > s["start"]]
    if not ref_proc:
        return {
            "der": None,
            "false_alarm_sec": 0.0,
            "miss_sec": 0.0,
            "confusion_sec": 0.0,
            "scored_ref_time_sec": 0.0,
            "ignored_ref_overlap_time_sec": 0.0,
            "method": "approx_interval_single_label_nooverlap",
            "collar_sec": collar_sec,
        }

    boundaries = sorted(
        {
            *[float(s["start"]) for s in ref_proc],
            *[float(s["end"]) for s in ref_proc],
            *[float(s["start"]) for s in hyp_proc],
            *[float(s["end"]) for s in hyp_proc],
        }
    )
    if len(boundaries) < 2:
        return {
            "der": None,
            "false_alarm_sec": 0.0,
            "miss_sec": 0.0,
            "confusion_sec": 0.0,
            "scored_ref_time_sec": 0.0,
            "ignored_ref_overlap_time_sec": 0.0,
            "method": "approx_interval_single_label_nooverlap",
            "collar_sec": collar_sec,
        }

    atoms = _atomize(boundaries, ref_proc, hyp_proc)
    hyp_to_ref = _best_hyp_to_ref_mapping(atoms)

    fa = 0.0
    miss = 0.0
    conf = 0.0
    scored_ref = 0.0
    ignored_overlap = 0.0

    for atom in atoms:
        dur = atom["dur"]
        ref_active = atom["ref"]
        hyp_active = atom["hyp"]
        if skip_overlap and len(ref_active) > 1:
            ignored_overlap += dur
            continue

        if len(ref_active) == 0:
            if hyp_active:
                fa += dur * len(hyp_active)
            continue

        # Single-ref scoring path (skip_overlap=True typical).
        scored_ref += dur
        ref_label = sorted(ref_active)[0]
        if not hyp_active:
            miss += dur
            continue

        mapped_matches = any(hyp_to_ref.get(h) == ref_label for h in hyp_active)
        if mapped_matches:
            # Extra simultaneous hypotheses beyond one are counted as false alarm speaker-time.
            if len(hyp_active) > 1:
                fa += dur * (len(hyp_active) - 1)
        else:
            conf += dur
            if len(hyp_active) > 1:
                fa += dur * (len(hyp_active) - 1)

    der = None if scored_ref <= 0 else (fa + miss + conf) / scored_ref
    return {
        "der": der,
        "false_alarm_sec": round(fa, 6),
        "miss_sec": round(miss, 6),
        "confusion_sec": round(conf, 6),
        "scored_ref_time_sec": round(scored_ref, 6),
        "ignored_ref_overlap_time_sec": round(ignored_overlap, 6),
        "method": "approx_interval_single_label_nooverlap",
        "collar_sec": collar_sec,
        "mapping": [{"hyp_speaker": h, "ref_speaker": r} for h, r in sorted(hyp_to_ref.items())],
    }


def _apply_collar(segments: list[dict[str, Any]], collar_sec: float) -> list[dict[str, Any]]:
    """Trim segment boundaries by the requested collar before DER scoring."""
    out: list[dict[str, Any]] = []
    for s in segments:
        start = float(s["start"]) + collar_sec
        end = float(s["end"]) - collar_sec
        if end <= start:
            continue
        out.append({"start": start, "end": end, "speaker": str(s["speaker"])})
    return out


def _atomize(
    boundaries: list[float],
    ref_segments: list[dict[str, Any]],
    hyp_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split the timeline into atomic intervals spanning all boundaries."""
    atoms: list[dict[str, Any]] = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        ref_active = {
            str(s["speaker"])
            for s in ref_segments
            if float(s["start"]) <= mid < float(s["end"])
        }
        hyp_active = {
            str(s["speaker"])
            for s in hyp_segments
            if float(s["start"]) <= mid < float(s["end"])
        }
        atoms.append({"start": a, "end": b, "dur": b - a, "ref": ref_active, "hyp": hyp_active})
    return atoms


def _best_hyp_to_ref_mapping(atoms: list[dict[str, Any]]) -> dict[str, str]:
    """Find the best speaker mapping for the approximate DER computation."""
    overlap: dict[tuple[str, str], float] = {}
    hyp_labels: set[str] = set()
    ref_labels: set[str] = set()
    for atom in atoms:
        if len(atom["ref"]) != 1 or len(atom["hyp"]) != 1:
            continue
        ref = next(iter(atom["ref"]))
        hyp = next(iter(atom["hyp"]))
        hyp_labels.add(hyp)
        ref_labels.add(ref)
        overlap[(hyp, ref)] = overlap.get((hyp, ref), 0.0) + float(atom["dur"])

    hyp_list = sorted(hyp_labels)
    ref_list = sorted(ref_labels)
    if not hyp_list or not ref_list:
        return {}

    n = max(len(hyp_list), len(ref_list))
    if n > 8:
        # Fallback greedy for unusual speaker counts.
        pairs = sorted(overlap.items(), key=lambda kv: kv[1], reverse=True)
        used_h: set[str] = set()
        used_r: set[str] = set()
        out: dict[str, str] = {}
        for (h, r), _w in pairs:
            if h in used_h or r in used_r:
                continue
            out[h] = r
            used_h.add(h)
            used_r.add(r)
        return out

    hyp_pad = hyp_list + [f"__hyp_dummy_{i}" for i in range(n - len(hyp_list))]
    ref_pad = ref_list + [f"__ref_dummy_{i}" for i in range(n - len(ref_list))]

    weights = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, h in enumerate(hyp_pad):
        for j, r in enumerate(ref_pad):
            if h.startswith("__hyp_dummy_") or r.startswith("__ref_dummy_"):
                weights[i][j] = 0.0
            else:
                weights[i][j] = overlap.get((h, r), 0.0)

    best_score: float | None = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(n)):
        score = 0.0
        for i, j in enumerate(perm):
            score += weights[i][j]
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm

    assert best_perm is not None
    out: dict[str, str] = {}
    for i, j in enumerate(best_perm):
        h = hyp_pad[i]
        r = ref_pad[j]
        if h.startswith("__hyp_dummy_") or r.startswith("__ref_dummy_"):
            continue
        out[h] = r
    return out
