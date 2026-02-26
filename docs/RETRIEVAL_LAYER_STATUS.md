# Retrieval Layer Status (Optional Plan Component)

## Status

`Deferred (Optional)` for current milestone / acceptance target.

The plan explicitly marks retrieval as optional. The current implementation does not require FAISS or sentence-transformers to satisfy core offline AMI pipeline goals.

## Why Deferred

Current limiting factors are:

- speech hypothesis quality (`WER`, `cpWER`, `DER`)
- diarization quality on harder meetings
- MoM extraction precision/recall tuning under noisy ASR

The project already implemented a lightweight non-embedding optimization:

- summary-guided hybrid chunk selection for extraction (`llama.cpp` backend)

This provides part of the retrieval benefit (reduced LLM calls and better precision) without adding FAISS / embedding model complexity.

## What Would a Minimal Retrieval Layer Look Like (Future)

Components:

- `sentence-transformers` (offline local embedding model)
- `FAISS` (local index)

Inputs:

- `artifacts/ami/{meeting_id}/transcript_chunks.jsonl`

Outputs (plan-aligned):

- `artifacts/ami/{meeting_id}/faiss_index/`
- `artifacts/ami/{meeting_id}/retrieval_results.json`

Integration points:

- summarization preselection for long meetings
- extraction evidence candidate selection
- audit UI / manual review support ("show supporting chunks")

## Recommended Trigger to Implement

Implement retrieval only if one or more of these occur:

1. LLM runtime on long meetings becomes the dominant bottleneck
2. Extraction recall remains low after transcript quality improvements
3. You want a stronger research contribution around retrieval-augmented MoM generation

## Current Plan Alignment Position

For the current capstone milestone:

- retrieval remains `Pending` / `Optional`
- this does **not** block Section 16 acceptance if other mandatory criteria are met
