# Transcript Normalization Decision (spaCy vs Rule-Based)

## Decision Summary

Current implementation uses **rule-based transcript normalization** as the primary normalization path.

Status relative to plan:

- Plan component list mentions `spaCy (offline English model)` as a candidate component.
- Current implementation intentionally prioritizes rule-based normalization for offline robustness, simplicity, and deterministic behavior.

## Why Rule-Based Was Chosen (Current Default)

1. Offline reliability
- No additional runtime model downloads required
- Fewer environment issues on constrained/offline setups

2. Determinism and auditability
- Rules are easier to inspect and version
- Fewer hidden model-behavior changes across environments

3. Performance on GTX 1080 Ti workstation context
- CPU overhead remains low
- Keeps preprocessing lightweight relative to NeMo and `llama.cpp` stages

4. Good enough for current pipeline focus
- Current quality bottleneck is primarily speech hypothesis quality (`WER/cpWER/DER`), not normalization sophistication

## What Is Implemented

- Raw transcript view preserved:
  - `artifacts/ami/{meeting_id}/transcript_raw.json`
- Normalized transcript view preserved:
  - `artifacts/ami/{meeting_id}/transcript_normalized.json`
- Canonical Pydantic meeting object built from normalized text:
  - `artifacts/ami/meetings_canonical.jsonl`

Code references:

- `src/ami_mom_pipeline/pipeline.py`
- `src/ami_mom_pipeline/schemas/models.py`

## When to Add spaCy (Optional Upgrade)

Add offline spaCy normalization if one of these becomes a priority:

- stronger sentence segmentation / entity-aware normalization
- improved downstream extraction recall from cleaner punctuation/casing
- explicit academic requirement to demonstrate model-based NLP preprocessing

If added later:

- keep it optional behind config
- preserve current rule-based path as deterministic fallback
- pin offline model artifact and document checksum/version

## Recommended Position in Thesis/Report

Document this as:

- a deliberate engineering tradeoff (offline reproducibility over preprocessing complexity)
- justified by current bottleneck analysis (speech recognition quality dominates MoM quality)

This keeps plan alignment strong while remaining honest and defensible.
