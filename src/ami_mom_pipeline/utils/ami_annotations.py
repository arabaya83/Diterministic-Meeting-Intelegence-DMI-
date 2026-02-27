"""AMI annotation parsing utilities.

This module loads AMI XML word-level annotations and converts them into
deterministic token/utterance structures consumed by VAD/diarization mock
fallbacks and evaluation helpers.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _localname(tag: str) -> str:
    """Return XML local tag name without namespace prefix."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def load_word_tokens(annotations_dir: Path, meeting_id: str) -> list[dict[str, Any]]:
    """Load AMI word tokens for a meeting from XML annotations.

    Args:
        annotations_dir: Root annotation directory containing `words/`.
        meeting_id: AMI meeting identifier.

    Returns:
        list[dict[str, Any]]: Time-sorted token rows with speaker letter,
        text, and punctuation flags.
    """
    words_dir = annotations_dir / "words"
    pattern = f"{meeting_id}.*.words.xml"
    tokens: list[dict] = []
    for path in sorted(words_dir.glob(pattern)):
        m = re.match(rf"{re.escape(meeting_id)}\.([A-Z])\.words\.xml$", path.name)
        if not m:
            continue
        speaker_letter = m.group(1)
        root = ET.parse(path).getroot()
        for elem in root:
            tag = _localname(elem.tag)
            if tag != "w":
                continue
            start = elem.attrib.get("starttime")
            end = elem.attrib.get("endtime")
            if start is None or end is None:
                continue
            text = html.unescape((elem.text or "").strip())
            if not text:
                continue
            tokens.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "speaker_letter": speaker_letter,
                    "text": text,
                    "is_punc": elem.attrib.get("punc") == "true",
                }
            )
    # Stable temporal + speaker sort is required for deterministic utterance
    # grouping and reproducible reference text generation.
    tokens.sort(key=lambda t: (t["start"], t["end"], t["speaker_letter"]))
    return tokens


def build_utterances(
    tokens: list[dict[str, Any]],
    max_pause_sec: float = 1.2,
    max_dur_sec: float = 30.0,
) -> list[dict[str, Any]]:
    """Group word tokens into speaker-homogeneous utterances.

    Split rules:
    - speaker change
    - pause larger than `max_pause_sec`
    - utterance duration reaching `max_dur_sec`
    """
    utterances: list[dict] = []
    if not tokens:
        return utterances

    cur: dict | None = None
    for tok in tokens:
        if cur is None:
            cur = {
                "speaker_letter": tok["speaker_letter"],
                "start": tok["start"],
                "end": tok["end"],
                "tokens": [tok],
            }
            continue
        pause = tok["start"] - cur["end"]
        cur_dur = cur["end"] - cur["start"]
        split = (
            tok["speaker_letter"] != cur["speaker_letter"]
            or pause > max_pause_sec
            or cur_dur >= max_dur_sec
        )
        if split:
            utterances.append(_finalize_utterance(cur))
            cur = {
                "speaker_letter": tok["speaker_letter"],
                "start": tok["start"],
                "end": tok["end"],
                "tokens": [tok],
            }
        else:
            cur["tokens"].append(tok)
            cur["end"] = max(cur["end"], tok["end"])
    if cur is not None:
        utterances.append(_finalize_utterance(cur))
    return utterances


def _finalize_utterance(cur: dict[str, Any]) -> dict[str, Any]:
    """Convert in-progress utterance token buffer into persisted row."""
    pieces: list[str] = []
    for tok in cur["tokens"]:
        if tok["is_punc"] and pieces:
            pieces[-1] = pieces[-1] + tok["text"]
        else:
            pieces.append(tok["text"])
    text = " ".join(pieces).strip()
    return {
        "speaker_letter": cur["speaker_letter"],
        "start": round(float(cur["start"]), 3),
        "end": round(float(cur["end"]), 3),
        "text": text,
        "token_count": sum(0 if t["is_punc"] else 1 for t in cur["tokens"]),
    }


def reference_plain_text(tokens: list[dict[str, Any]]) -> str:
    """Build flattened reference transcript text from AMI tokens."""
    pieces: list[str] = []
    for tok in tokens:
        if tok["is_punc"] and pieces:
            pieces[-1] = pieces[-1] + tok["text"]
        else:
            pieces.append(tok["text"])
    return " ".join(pieces).strip()
