"""Pydantic configuration models for the AMI pipeline.

This module defines YAML-backed runtime configuration for:
- stage controls (chunking, speech backend, summarization/extraction backends)
- path layout
- offline/reproducibility/runtime guardrails
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    """Chunking parameters for transcript segmentation."""

    target_words: int = 220
    overlap_words: int = 40


class NemoConfig(BaseModel):
    """Local NeMo backend configuration and command templates."""

    vad_model_path: str | None = None
    diarizer_config_path: str | None = None
    asr_model_path: str | None = None
    vad_command: str | None = None
    diarization_command: str | None = None
    asr_command: str | None = None
    allow_precomputed_outputs: bool = True


class SpeechBackendConfig(BaseModel):
    """Speech backend selector and backend-specific config."""

    mode: Literal["mock", "nemo"] = "mock"
    nemo: NemoConfig = Field(default_factory=NemoConfig)


class LlamaCppConfig(BaseModel):
    """Local llama.cpp model/runtime configuration."""

    model_path: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = 20
    temperature: float = 0.05
    top_p: float = 1.0
    repeat_penalty: float = 1.05


class SummarizationBackendConfig(BaseModel):
    """Summarization backend selector and config."""

    mode: Literal["mock", "llama_cpp"] = "mock"
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)


class ExtractionBackendConfig(BaseModel):
    """Extraction backend selector and config."""

    mode: Literal["mock", "llama_cpp"] = "mock"
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)


class PipelineSettings(BaseModel):
    """Top-level stage settings used by pipeline orchestration."""

    seed: int = 42
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    speech_backend: SpeechBackendConfig = Field(default_factory=SpeechBackendConfig)
    summarization_backend: SummarizationBackendConfig = Field(default_factory=SummarizationBackendConfig)
    extraction_backend: ExtractionBackendConfig = Field(default_factory=ExtractionBackendConfig)


class PathsConfig(BaseModel):
    """Filesystem path settings for raw/staged/artifact/model locations."""

    raw_audio_dir: str = "data/rawa/ami/audio"
    annotations_dir: str = "data/rawa/ami/annotations"
    staged_dir: str = "data/staged/ami"
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"


class RuntimeConfig(BaseModel):
    """Runtime controls for offline enforcement and reproducibility."""

    offline: bool = True
    fail_on_missing_models: bool = False
    overwrite: bool = True
    deterministic_mode: bool = True
    fail_on_determinism_risks: bool = False
    write_stage_trace: bool = True
    write_preflight_audit: bool = True
    fail_on_offline_violations: bool = True
    include_nondeterministic_timings_in_manifest: bool = False
    enable_mlflow_logging: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "ami_mom_offline"


class AppConfig(BaseModel):
    """Root application config loaded from YAML."""

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | None = None) -> "AppConfig":
        """Load config from YAML file path or return defaults when missing."""
        if path is None:
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def resolve_path(self, relative_or_abs: str) -> Path:
        """Resolve a path string to expanded `Path` instance."""
        return Path(relative_or_abs).expanduser()
