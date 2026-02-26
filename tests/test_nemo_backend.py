from pathlib import Path

import pytest

from ami_mom_pipeline.backends.nemo_backend import NemoBackendError, NemoSpeechBackend
from ami_mom_pipeline.config import AppConfig


def test_nemo_backend_rejects_url_model_path():
    cfg = AppConfig.load(None)
    cfg.pipeline.speech_backend.mode = "nemo"
    cfg.pipeline.speech_backend.nemo.vad_model_path = "https://example.com/model.nemo"
    backend = NemoSpeechBackend(cfg)
    with pytest.raises(NemoBackendError, match="Offline mode requires local paths"):
        backend.run_vad("ES2005a", Path("audio.wav"), Path("artifacts/ami/ES2005a"))


def test_nemo_backend_requires_command_or_precomputed(tmp_path: Path):
    cfg = AppConfig.load(None)
    cfg.pipeline.speech_backend.mode = "nemo"
    cfg.pipeline.speech_backend.nemo.vad_model_path = str(tmp_path / "models" / "vad")
    cfg.runtime.fail_on_missing_models = False
    cfg.pipeline.speech_backend.nemo.allow_precomputed_outputs = False
    backend = NemoSpeechBackend(cfg)
    with pytest.raises(NemoBackendError, match="no 'vad_command' configured"):
        backend.run_vad("ES2005a", tmp_path / "audio.wav", tmp_path / "out")
