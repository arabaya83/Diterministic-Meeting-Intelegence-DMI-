# Local Model Artifacts (Offline Runtime)

Place one-time downloaded model artifacts here so the pipeline can run fully offline.

## Required for NeMo speech stack

- `models/nemo/vad/`
  - local NeMo VAD model(s) used by your pinned VAD runner
- `models/nemo/diarizer/`
  - diarizer config YAML (for example `inference.yaml`)
  - any local model/config references required by that YAML
- `models/nemo/asr/`
  - local ASR `.nemo` file (recommended)

## Optional

- `models/llm/gguf/`
- `models/embeddings/`

## Validation

Run:

```bash
source scripts/env_offline.sh
python scripts/check_nemo_models.py --config configs/pipeline.nemo.yaml
```

