from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ..config import AppConfig
from ..schemas.models import ASRSegment, DiarizationSegment, VADSegment


class NemoBackendError(RuntimeError):
    pass


@dataclass
class NemoOutputs:
    vad_json: Path
    vad_rttm: Path
    diarization_json: Path
    diarization_rttm: Path
    asr_json: Path
    asr_conf_json: Path
    full_transcript_txt: Path


class NemoSpeechBackend:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.nemo_cfg = cfg.pipeline.speech_backend.nemo

    def run_vad(self, meeting_id: str, audio_path: Path, output_dir: Path) -> dict[str, Any]:
        self._validate_offline_assets(require=("vad_model_path",))
        outputs = self._outputs(output_dir)
        if self._can_reuse_vad(outputs):
            return self._load_vad_outputs(outputs)

        cmd_template = self.nemo_cfg.vad_command
        if not cmd_template:
            raise NemoBackendError(
                "NeMo VAD requested but no reusable outputs found and no 'vad_command' configured. "
                "Provide pipeline.speech_backend.nemo.vad_command or precompute "
                f"'{outputs.vad_json.name}'/'{outputs.vad_rttm.name}' in {output_dir}."
            )
        self._run_command(
            cmd_template,
            meeting_id=meeting_id,
            audio_path=audio_path,
            output_dir=output_dir,
        )
        return self._load_vad_outputs(outputs)

    def run_diarization(self, meeting_id: str, audio_path: Path, output_dir: Path) -> dict[str, Any]:
        self._validate_offline_assets(require=("diarizer_config_path",))
        outputs = self._outputs(output_dir)
        if self._can_reuse_diarization(outputs):
            return self._load_diarization_outputs(outputs)

        cmd_template = self.nemo_cfg.diarization_command
        if not cmd_template:
            raise NemoBackendError(
                "NeMo diarization requested but no reusable outputs found and no 'diarization_command' configured. "
                "Provide pipeline.speech_backend.nemo.diarization_command or precompute "
                f"'{outputs.diarization_json.name}'/'{outputs.diarization_rttm.name}' in {output_dir}."
            )
        self._run_command(
            cmd_template,
            meeting_id=meeting_id,
            audio_path=audio_path,
            output_dir=output_dir,
        )
        return self._load_diarization_outputs(outputs)

    def run_asr(
        self,
        meeting_id: str,
        audio_path: Path,
        output_dir: Path,
        diarization_segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._validate_offline_assets(require=("asr_model_path",))
        outputs = self._outputs(output_dir)
        if self._can_reuse_asr(outputs):
            return self._load_asr_outputs(outputs)

        cmd_template = self.nemo_cfg.asr_command
        if not cmd_template:
            raise NemoBackendError(
                "NeMo ASR requested but no reusable outputs found and no 'asr_command' configured. "
                "Provide pipeline.speech_backend.nemo.asr_command or precompute "
                f"'{outputs.asr_json.name}', '{outputs.full_transcript_txt.name}', and '{outputs.asr_conf_json.name}' in {output_dir}."
            )

        diar_path = outputs.diarization_json if outputs.diarization_json.exists() else None
        self._run_command(
            cmd_template,
            meeting_id=meeting_id,
            audio_path=audio_path,
            output_dir=output_dir,
            diarization_json=diar_path,
        )
        return self._load_asr_outputs(outputs)

    def _outputs(self, output_dir: Path) -> NemoOutputs:
        return NemoOutputs(
            vad_json=output_dir / "vad_segments.json",
            vad_rttm=output_dir / "vad_segments.rttm",
            diarization_json=output_dir / "diarization_segments.json",
            diarization_rttm=output_dir / "diarization.rttm",
            asr_json=output_dir / "asr_segments.json",
            asr_conf_json=output_dir / "asr_confidence.json",
            full_transcript_txt=output_dir / "full_transcript.txt",
        )

    def _validate_offline_assets(self, require: tuple[str, ...]) -> None:
        for field_name in require:
            value = getattr(self.nemo_cfg, field_name)
            if not value:
                raise NemoBackendError(f"Missing NeMo config field: pipeline.speech_backend.nemo.{field_name}")
            self._ensure_local_path(field_name, value)

    def _ensure_local_path(self, field_name: str, value: str) -> Path:
        if "://" in value:
            raise NemoBackendError(
                f"Offline mode requires local paths, but '{field_name}' looks like a URL: {value}"
            )
        path = Path(value).expanduser()
        if self.cfg.runtime.fail_on_missing_models and not path.exists():
            raise NemoBackendError(f"Configured NeMo asset path does not exist: {path}")
        return path

    def _can_reuse(self, *paths: Path) -> bool:
        if not self.nemo_cfg.allow_precomputed_outputs:
            return False
        return all(p.exists() for p in paths)

    def _can_reuse_vad(self, outputs: NemoOutputs) -> bool:
        if not self._can_reuse(outputs.vad_json, outputs.vad_rttm):
            return False
        try:
            segs = self._read_json(outputs.vad_json)
        except Exception:
            return False
        if not isinstance(segs, list):
            return False
        allowed_prefixes = ("nemo",)
        for s in segs:
            src = str(s.get("source", ""))
            if src and not src.startswith(allowed_prefixes):
                return False
        return True

    def _can_reuse_diarization(self, outputs: NemoOutputs) -> bool:
        if not self._can_reuse(outputs.diarization_json, outputs.diarization_rttm):
            return False
        try:
            segs = self._read_json(outputs.diarization_json)
        except Exception:
            return False
        if not isinstance(segs, list):
            return False
        for s in segs:
            src = str(s.get("source", ""))
            if src and not src.startswith("nemo"):
                return False
        return True

    def _can_reuse_asr(self, outputs: NemoOutputs) -> bool:
        if not self._can_reuse(outputs.asr_json, outputs.full_transcript_txt, outputs.asr_conf_json):
            return False
        try:
            segs = self._read_json(outputs.asr_json)
        except Exception:
            return False
        if not isinstance(segs, list):
            return False
        for s in segs:
            src = str(s.get("source", ""))
            if src and not src.startswith("nemo"):
                return False
        return True

    def _run_command(self, template: str, **kwargs: Any) -> None:
        fmt = {
            "meeting_id": kwargs.get("meeting_id"),
            "audio_path": str(kwargs.get("audio_path")) if kwargs.get("audio_path") else "",
            "output_dir": str(kwargs.get("output_dir")) if kwargs.get("output_dir") else "",
            "vad_model_path": self.nemo_cfg.vad_model_path or "",
            "diarizer_config_path": self.nemo_cfg.diarizer_config_path or "",
            "asr_model_path": self.nemo_cfg.asr_model_path or "",
            "diarization_json": str(kwargs.get("diarization_json")) if kwargs.get("diarization_json") else "",
        }
        try:
            command = template.format(**fmt)
        except KeyError as e:
            raise NemoBackendError(f"Unknown placeholder in NeMo command template: {e}") from e
        if not command.strip():
            raise NemoBackendError("NeMo command template rendered to an empty command.")
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise NemoBackendError(
                "NeMo command failed with non-zero exit code "
                f"{proc.returncode}. stdout='{stdout[:500]}' stderr='{stderr[:500]}'"
            )

    def _load_vad_outputs(self, outputs: NemoOutputs) -> dict[str, Any]:
        segments = self._read_json(outputs.vad_json)
        TypeAdapter(list[VADSegment]).validate_python(segments)
        return {"count": len(segments), "segments": segments}

    def _load_diarization_outputs(self, outputs: NemoOutputs) -> dict[str, Any]:
        segments = self._read_json(outputs.diarization_json)
        TypeAdapter(list[DiarizationSegment]).validate_python(segments)
        speakers = sorted({s["speaker"] for s in segments})
        return {"count": len(segments), "segments": segments, "speaker_labels": speakers}

    def _load_asr_outputs(self, outputs: NemoOutputs) -> dict[str, Any]:
        segments = self._read_json(outputs.asr_json)
        TypeAdapter(list[ASRSegment]).validate_python(segments)
        conf = self._read_json(outputs.asr_conf_json)
        return {"segments": segments, "confidence": conf}

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise NemoBackendError(f"Expected NeMo output not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
