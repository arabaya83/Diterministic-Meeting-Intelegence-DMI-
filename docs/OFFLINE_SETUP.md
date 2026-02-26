# Offline Setup Notes

This scaffold is offline-first after one-time artifact acquisition.

Expected local directories:

- `models/nemo/` (VAD/diarization/ASR models and configs)
- `models/llm/gguf/` (quantized instruct GGUF models for llama.cpp)
- `models/embeddings/` (optional sentence-transformer local models)
- `wheelhouse/` (optional local wheel cache)
- `environment_lockfiles/` (pip/conda lockfiles)

## Integration points

- `src/ami_mom_pipeline/backends/nemo_backend.py`
- `src/ami_mom_pipeline/backends/llama_cpp_backend.py`

`NemoSpeechBackend` is implemented as an offline local adapter that can:

- reuse precomputed NeMo outputs already written into `artifacts/ami/{meeting_id}/`
- run local NeMo scripts via configured command templates and then validate/load outputs

Example `configs/pipeline.sample.yaml` overrides:

```yaml
pipeline:
  speech_backend:
    mode: nemo
    nemo:
      vad_model_path: models/nemo/vad/model.nemo
      diarizer_config_path: models/nemo/diarizer/inference.yaml
      asr_model_path: models/nemo/asr/conformer_ctc.nemo
      vad_command: >
        python scripts/nemo_vad.py
        --audio {audio_path}
        --out-dir {output_dir}
        --meeting-id {meeting_id}
        --delegate-cmd "python path/to/pinned_nemo_vad_runner.py --audio {audio_path} --model {vad_model_path} --out-dir {output_dir}"
      diarization_command: >
        python scripts/nemo_diarize.py
        --audio {audio_path}
        --config {diarizer_config_path}
        --out-dir {output_dir}
        --meeting-id {meeting_id}
        --try-nemo-api
      asr_command: >
        python scripts/nemo_asr.py
        --audio {audio_path}
        --model {asr_model_path}
        --diarization-json {diarization_json}
        --out-dir {output_dir}
        --meeting-id {meeting_id}
        --try-nemo-api
```

Expected output files produced (or precomputed) in `artifacts/ami/{meeting_id}/`:

- `vad_segments.json`
- `vad_segments.rttm`
- `diarization_segments.json`
- `diarization.rttm`
- `asr_segments.json`
- `asr_confidence.json`
- `full_transcript.txt`

Notes:

- `scripts/nemo_diarize.py` and `scripts/nemo_asr.py` include best-effort NeMo API execution paths (`--try-nemo-api`), but exact compatibility depends on your pinned NeMo version/config schema.
- `scripts/nemo_vad.py` is a stable artifact-contract wrapper and normalization step; use `--delegate-cmd` to call your pinned NeMo VAD runner or pass precomputed VAD outputs.

## Offline compliance controls (implemented)

Runtime flags (in config `runtime:`):

- `offline: true`
- `write_preflight_audit: true`
- `fail_on_offline_violations: true`
- `fail_on_missing_models: true` (recommended for strict offline runs)

Per-meeting artifact:

- `artifacts/ami/{meeting_id}/preflight_offline_audit.json`

The preflight audit checks:

- local-only model/config paths (no URL model paths)
- NeMo command templates for explicit URL/download usage (`http(s)`, `curl`, `wget`)
- offline-related environment flag visibility (warning-level)

## Reproducibility / traceability artifacts

Per meeting:

- `reproducibility_report.json` (config digest, environment snapshot, determinism settings, code hashes)
- `stage_trace.jsonl` (stage timings + status events)
- `run_manifest.json` (compact deterministic artifact summary + digest)
