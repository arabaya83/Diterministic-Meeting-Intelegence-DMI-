"""AMI annotation parsing utilities.

This module is the bridge between the frozen AMI annotation tree in
`data/rawa/ami/annotations/` and the rest of the pipeline. It provides:

- deterministic loading of word-level reference tokens used for transcript
  reconstruction and ASR evaluation
- utterance grouping helpers used by mock/fallback speech stages
- access to AMI abstractive summary references (`*.abssumm.xml`) used for
  ROUGE evaluation
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


def load_abstractive_summary_sections(annotations_dir: Path, meeting_id: str) -> dict[str, list[str]]:
    """Load AMI abstractive-summary reference sections for a meeting.

    The AMI abstractive files expose four top-level buckets that are useful
    both for evaluation and prompt design: `abstract`, `actions`,
    `decisions`, and `problems`.

    Returns:
        dict[str, list[str]]: Section-to-sentences mapping. Missing files
        return empty sections instead of raising so evaluation can remain
        reference-optional for meetings that do not have annotations.
    """
    path = annotations_dir / "abstractive" / f"{meeting_id}.abssumm.xml"
    sections = {"abstract": [], "actions": [], "decisions": [], "problems": []}
    if not path.exists():
        return sections

    root = ET.parse(path).getroot()
    for child in root:
        section = _localname(child.tag)
        if section not in sections:
            continue
        sentences: list[str] = []
        for sentence in child:
            if _localname(sentence.tag) != "sentence":
                continue
            text = html.unescape(" ".join((sentence.text or "").split()))
            if text:
                sentences.append(text)
        sections[section] = sentences
    return sections


def load_abstractive_summary_text(annotations_dir: Path, meeting_id: str, section: str = "abstract") -> str:
    """Load flattened AMI abstractive reference text for one section.

    This is used by the evaluation stage to compute ROUGE against the AMI
    human-written abstractive summary. The `abstract` section is the default
    because it aligns best with the narrative `summary` field in
    `mom_summary.json`.
    """
    sections = load_abstractive_summary_sections(annotations_dir, meeting_id)
    sentences = sections.get(section, [])
    return " ".join(sentences).strip()
