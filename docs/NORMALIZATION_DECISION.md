# Transcript Normalization Decision (spaCy vs Rule-Based)

## Decision Summary

Current implementation supports **spaCy-backed normalization with an offline-safe fallback**.

Status relative to plan:

- Plan component list mentions `spaCy (offline English model)` and rule-based normalization.
- The pipeline now supports both, with configurable behavior:
  - `mode: spacy` uses `en_core_web_sm` when available offline
  - automatic fallback to `spacy.blank("en")` if the model package is unavailable
  - `mode: rule` remains available for strict rule-only normalization

## Why This Hybrid Approach Was Chosen

1. Offline reliability
- No runtime network downloads required
- Uses local spaCy model when present, otherwise local `spacy.blank("en")`

2. Determinism and auditability
- Normalization mode used is recorded in canonical meeting metadata
- Behavior is explicit and config-driven

3. Performance on GTX 1080 Ti workstation context
- CPU overhead remains low
- Keeps preprocessing lightweight relative to NeMo and `llama.cpp` stages

4. Good enough for current pipeline focus
- Current quality bottleneck remains primarily speech hypothesis quality (`WER/cpWER/DER`)

## What Is Implemented

- Raw transcript view preserved:
  - `artifacts/ami/{meeting_id}/transcript_raw.json`
- Normalized transcript view preserved (spaCy/rule mode):
  - `artifacts/ami/{meeting_id}/transcript_normalized.json`
- Canonical Pydantic meeting object built from normalized text:
  - `artifacts/ami/meetings_canonical.jsonl`

Code references:

- `src/ami_mom_pipeline/pipeline.py` (`normalize_text`)
- `src/ami_mom_pipeline/config.py` (`pipeline.normalization`)
- `src/ami_mom_pipeline/schemas/models.py`

## Recommended settings

For final offline runs:

- `pipeline.normalization.mode: spacy`
- `pipeline.normalization.spacy_model: en_core_web_sm`
- `pipeline.normalization.fail_on_spacy_missing: false`

This gives model-based normalization when available while preserving offline robustness.
