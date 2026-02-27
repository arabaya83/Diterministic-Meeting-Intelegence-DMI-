# Retrieval Layer Status (Optional Plan Component)

## Status

`Implemented (Optional)` for current milestone / acceptance target.

The plan marks retrieval as optional. It is now implemented as an optional stage with an offline-safe default path.

## Current Implementation

Current limiting factors are:

- speech hypothesis quality (`WER`, `cpWER`, `DER`)
- diarization quality on harder meetings
- MoM extraction precision/recall tuning under noisy ASR

- offline lexical retrieval stage is always available (no extra dependencies)
- optional FAISS + sentence-transformers path is available when local dependencies/models are present
- retrieval artifacts are produced at:
  - `artifacts/ami/{meeting_id}/retrieval_results.json`
  - `artifacts/ami/{meeting_id}/faiss_index/` (when FAISS mode is enabled)

Code references:

- `src/ami_mom_pipeline/pipeline.py` (`stage_retrieval`)
- `src/ami_mom_pipeline/config.py` (`pipeline.retrieval.*`)
- `scripts/run_nemo_batch_sequential.py` (validation expects retrieval artifact when enabled)

## Notes

- Retrieval remains optional from a plan perspective, but it is no longer pending.
- The default production configs keep `use_faiss: false` for maximum offline portability.
