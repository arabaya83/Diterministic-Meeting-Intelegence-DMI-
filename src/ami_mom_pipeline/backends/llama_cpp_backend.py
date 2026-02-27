"""Local llama.cpp backend for offline summarization and extraction.

The backend enforces local model loading and produces schema-validated
`MinutesSummary` and `ExtractionOutput` objects used by downstream artifact
writers in `pipeline.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..schemas.models import ActionItem, DecisionItem, EvidenceBackedPoint, ExtractionOutput, MinutesSummary


class LlamaCppBackendError(RuntimeError):
    """Raised when llama backend configuration or generation fails."""

    pass


class LlamaCppBackend:
    """Offline wrapper around `llama-cpp-python` inference calls."""

    def __init__(self, cfg: AppConfig):
        """Initialize backend with application configuration."""
        self.cfg = cfg
        self._llm = None

    def summarize(self, meeting_id: str, turns: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> MinutesSummary:
        """Generate structured minutes summary from meeting chunks.

        Args:
            meeting_id: AMI meeting id.
            turns: Canonical transcript turns.
            chunks: Chunked transcript rows.

        Returns:
            MinutesSummary: Summary object with evidence-backed sections.
        """
        llm = self._get_llm()
        if not chunks:
            return MinutesSummary(
                meeting_id=meeting_id,
                summary=f"Meeting {meeting_id} contains no transcript chunks.",
                key_points=[],
                discussion_points=[],
                follow_up=[],
                prompt_template_version="llama-cpp-v1",
                backend="llama_cpp",
            )

        chunk_summaries: list[str] = []
        # Hierarchical summarization for long meetings.
        if len(chunks) > 3:
            for chunk in chunks:
                prompt = self._chunk_prompt(meeting_id, chunk)
                raw = self._generate(llm, prompt, max_tokens=220, task="summarization")
                parsed = self._parse_summary_json(raw)
                if parsed:
                    text = parsed.get("summary", "") or ""
                    if parsed.get("key_points"):
                        text = text + "\n" + "\n".join(f"- {p}" for p in parsed["key_points"][:3])
                else:
                    text = raw.strip()
                if text:
                    chunk_summaries.append(f"[{chunk['chunk_id']}] {text}")
        else:
            chunk_summaries = [f"[{c['chunk_id']}] {c['text']}" for c in chunks]

        final_prompt = self._meeting_prompt(meeting_id, chunk_summaries, turns_count=len(turns), chunk_count=len(chunks))
        raw_final = self._generate(llm, final_prompt, max_tokens=520, task="summarization")
        parsed_final = self._parse_summary_json(raw_final)

        if parsed_final:
            summary_text = parsed_final.get("summary") or f"Meeting {meeting_id} summarized with llama.cpp."
            key_points = [str(x).strip() for x in parsed_final.get("key_points", []) if str(x).strip()][:5]
            discussion_texts = [str(x).strip() for x in parsed_final.get("discussion_points", []) if str(x).strip()][:8]
            follow_up_texts = [str(x).strip() for x in parsed_final.get("follow_up", []) if str(x).strip()][:8]
        else:
            # Safe fallback when model does not emit JSON.
            summary_text = raw_final.strip() or f"Meeting {meeting_id} summarized with llama.cpp (unparsed response)."
            key_points = self._fallback_key_points(summary_text)
            discussion_texts = key_points[:]
            follow_up_texts = []

        if not discussion_texts and key_points:
            discussion_texts = key_points[:]

        discussion_points = self._attach_summary_point_evidence(discussion_texts, chunks, kind="discussion")
        follow_up = self._attach_summary_point_evidence(follow_up_texts, chunks, kind="follow_up")
        discussion_points = self._filter_summary_points(discussion_points, kind="discussion")
        follow_up = self._filter_summary_points(follow_up, kind="follow_up")
        if not discussion_points and key_points:
            discussion_points = self._filter_summary_points(
                self._attach_summary_point_evidence(key_points, chunks, kind="discussion"),
                kind="discussion",
            )
        if not follow_up:
            follow_up = self._fallback_follow_up_items(
                key_points=key_points,
                discussion_texts=discussion_texts,
                chunks=chunks,
            )

        return MinutesSummary(
            meeting_id=meeting_id,
            summary=summary_text,
            key_points=key_points,
            discussion_points=discussion_points,
            follow_up=follow_up,
            prompt_template_version="llama-cpp-v1",
            backend="llama_cpp",
        )

    def extract(self, *args: Any, **kwargs: Any) -> ExtractionOutput:
        """Extract structured decisions/action items from transcript chunks."""
        meeting_id = kwargs["meeting_id"]
        chunks = kwargs["chunks"]
        summary = kwargs.get("summary") or {}
        llm = self._get_llm(task="extraction")

        decisions: list[DecisionItem] = []
        actions: list[ActionItem] = []
        flags: list[str] = []
        selected_chunks = self._select_extraction_chunks(chunks, summary)
        if len(selected_chunks) < len(chunks):
            flags.append(f"hybrid_chunk_selection:{len(selected_chunks)}/{len(chunks)}")

        for chunk in selected_chunks:
            prompt = self._extract_chunk_prompt(meeting_id, chunk)
            raw = self._generate(llm, prompt, max_tokens=380, task="extraction")
            parsed = self._parse_extraction_json(raw)
            if not parsed:
                continue
            for d in parsed.get("decisions", []):
                text = str(d.get("decision", "")).strip()
                if not text:
                    continue
                decisions.append(
                    DecisionItem(
                        decision=text[:240],
                        evidence_chunk_ids=[chunk["chunk_id"]],
                        evidence_snippets=[self._evidence_snippet(chunk.get("text", ""), text)],
                        confidence=self._bounded_conf(d.get("confidence"), default=0.65),
                        uncertain=bool(d.get("uncertain", False)),
                    )
                )
            for a in parsed.get("action_items", []):
                text = str(a.get("action", "")).strip()
                if not text:
                    continue
                owner = str(a.get("owner")).strip() if a.get("owner") not in (None, "") else None
                due = str(a.get("due_date")).strip() if a.get("due_date") not in (None, "") else None
                actions.append(
                    ActionItem(
                        action=text[:240],
                        owner=owner,
                        due_date=due,
                        evidence_chunk_ids=[chunk["chunk_id"]],
                        evidence_snippets=[self._evidence_snippet(chunk.get("text", ""), text)],
                        confidence=self._bounded_conf(a.get("confidence"), default=0.6),
                        uncertain=bool(a.get("uncertain", False)),
                    )
                )
            for f in parsed.get("flags", []):
                fs = str(f).strip()
                if fs:
                    flags.append(fs[:120])

        decisions = self._dedupe_decisions(decisions)
        actions = self._dedupe_actions(actions)
        decisions, dropped_decisions = self._filter_decisions(decisions)
        actions, dropped_actions = self._filter_actions(actions)
        if dropped_decisions:
            flags.append(f"dropped_low_quality_decisions:{dropped_decisions}")
        if dropped_actions:
            flags.append(f"dropped_low_quality_actions:{dropped_actions}")
        flags = self._normalize_extraction_flags(flags, has_decisions=bool(decisions), has_actions=bool(actions))
        if not decisions:
            flags.append("no_decisions_detected_by_llama_cpp")
        if not actions:
            flags.append("no_actions_detected_by_llama_cpp")
        return ExtractionOutput(meeting_id=meeting_id, decisions=decisions, action_items=actions, flags=sorted(set(flags)))

    def _get_llm(self, task: str = "summarization"):
        """Lazily load and cache llama model for summarize/extract tasks."""
        if self._llm is not None:
            return self._llm

        if task == "extraction":
            cfg_obj = self.cfg.pipeline.extraction_backend.llama_cpp
            # Reuse summarization model config if extraction config not explicitly set.
            if not cfg_obj.model_path:
                cfg_obj = self.cfg.pipeline.summarization_backend.llama_cpp
        else:
            cfg_obj = self.cfg.pipeline.summarization_backend.llama_cpp

        model_path = cfg_obj.model_path
        if not model_path:
            raise LlamaCppBackendError(f"Missing llama_cpp model_path for {task} backend.")
        path = Path(model_path).expanduser()
        if self.cfg.runtime.offline and "://" in model_path:
            raise LlamaCppBackendError("Offline runtime requires a local llama.cpp model_path (GGUF file).")
        if self.cfg.runtime.fail_on_missing_models and not path.exists():
            raise LlamaCppBackendError(f"llama.cpp GGUF model not found: {path}")

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise LlamaCppBackendError(
                "llama_cpp Python package is not available. Install llama-cpp-python in the offline environment."
            ) from e

        self._llm = Llama(
            model_path=str(path),
            n_ctx=int(cfg_obj.n_ctx),
            n_gpu_layers=int(cfg_obj.n_gpu_layers),
            verbose=False,
        )
        return self._llm

    def _generate(self, llm, prompt: str, max_tokens: int, task: str = "summarization") -> str:
        """Execute one llama generation call and return text payload."""
        if task == "extraction":
            cfg_obj = self.cfg.pipeline.extraction_backend.llama_cpp
            if not cfg_obj.model_path:
                cfg_obj = self.cfg.pipeline.summarization_backend.llama_cpp
        else:
            cfg_obj = self.cfg.pipeline.summarization_backend.llama_cpp
        resp = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=float(cfg_obj.temperature),
            top_p=float(cfg_obj.top_p),
            repeat_penalty=float(cfg_obj.repeat_penalty),
            stop=["</json>", "\n\nHuman:", "\n\nUser:"],
        )
        try:
            return str(resp["choices"][0]["text"])
        except Exception:
            return str(resp)

    @staticmethod
    def _chunk_prompt(meeting_id: str, chunk: dict[str, Any]) -> str:
        """Create chunk-level summarization prompt with strict JSON contract."""
        return (
            "You summarize meeting transcript chunks.\n"
            "Return ONLY JSON with keys: summary (string), key_points (array of 1-3 strings).\n"
            "Be concise and factual. No markdown.\n\n"
            f"Meeting ID: {meeting_id}\n"
            f"Chunk ID: {chunk.get('chunk_id')}\n"
            "Transcript chunk:\n"
            f"{chunk.get('text','')}\n\n"
            "JSON:"
        )

    @staticmethod
    def _meeting_prompt(meeting_id: str, chunk_summaries: list[str], turns_count: int, chunk_count: int) -> str:
        """Create meeting-level synthesis prompt with strict JSON contract."""
        body = "\n\n".join(chunk_summaries[:80])
        return (
            "You create Minutes of Meeting summaries from transcript evidence.\n"
            "Return ONLY JSON with keys:\n"
            '- "summary": short narrative paragraph\n'
            '- "discussion_points": array of 3-8 concise discussed points (strings)\n'
            '- "follow_up": array of follow-up items/questions/next checks (strings). Empty array if none.\n'
            '- "key_points": array of up to 5 strings (legacy concise highlights)\n'
            "Rules:\n"
            "1) discussion_points must describe what was discussed, not actions or decisions unless explicitly discussed as options.\n"
            "2) follow_up must contain only explicit next steps, open questions, or future checks mentioned in evidence.\n"
            "3) Do not invent owners or due dates in this summary JSON.\n"
            "4) If no explicit follow-up exists, return an empty follow_up array.\n"
            "Use neutral factual wording. No markdown, no extra text.\n\n"
            f"Meeting ID: {meeting_id}\n"
            f"Turn count: {turns_count}\n"
            f"Chunk count: {chunk_count}\n"
            "Evidence summaries:\n"
            f"{body}\n\n"
            "JSON:"
        )

    @staticmethod
    def _parse_summary_json(text: str) -> dict[str, Any] | None:
        """Parse and normalize summary JSON from possibly noisy model output."""
        if not text:
            return None
        stripped = text.strip()
        candidates: list[dict[str, Any]] = []

        # Direct parse first.
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass

        # Robustly scan for one or more JSON objects in noisy model output.
        candidates.extend(LlamaCppBackend._extract_json_objects(stripped))

        best: dict[str, Any] | None = None
        best_score = -1
        for obj in candidates:
            normalized = LlamaCppBackend._normalize_summary_obj(obj)
            if not normalized:
                continue
            score = LlamaCppBackend._score_summary_obj(normalized)
            if score >= best_score:
                best = normalized
                best_score = score
        return best

    @staticmethod
    def _fallback_key_points(summary_text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"[.;]\s+", summary_text) if p.strip()]
        return parts[:5]

    def _attach_summary_point_evidence(
        self,
        point_texts: list[str],
        chunks: list[dict[str, Any]],
        kind: str,
    ) -> list[EvidenceBackedPoint]:
        out: list[EvidenceBackedPoint] = []
        min_score = 2 if kind == "discussion" else 4
        for point_text in point_texts:
            text = self._clean_summary_text(point_text)
            if not text:
                continue
            chosen = self._best_evidence_chunks_for_point(text, chunks)
            if not chosen:
                continue
            if chosen[0][0] < min_score:
                continue
            chunk_ids = [str(c.get("chunk_id")) for _, c in chosen if c.get("chunk_id")]
            snippets = [self._evidence_snippet(str(c.get("text", "") or ""), text) for _, c in chosen]
            score = chosen[0][0] if chosen else 1
            out.append(
                EvidenceBackedPoint(
                    text=text[:240],
                    evidence_chunk_ids=chunk_ids[:3],
                    evidence_snippets=self._merge_snippets([], snippets),
                    confidence=round(min(0.95, max(0.2, score / 10.0)), 3),
                )
            )
        return out[:10]

    def _fallback_follow_up_items(
        self,
        key_points: list[str],
        discussion_texts: list[str],
        chunks: list[dict[str, Any]],
    ) -> list[EvidenceBackedPoint]:
        candidates: list[str] = []
        for text in list(discussion_texts) + list(key_points):
            t = self._clean_summary_text(text)
            if not t or t in candidates:
                continue
            candidates.append(t)
        if not candidates:
            return []
        prioritized = sorted(
            candidates,
            key=lambda x: (0 if self._looks_like_follow_up_candidate(x) else 1, -len(x)),
        )
        attached = self._attach_summary_point_evidence(prioritized[:6], chunks, kind="follow_up")
        filtered = self._filter_summary_points(attached, kind="follow_up")
        strong: list[EvidenceBackedPoint] = []
        for item in filtered:
            if item.confidence < 0.5:
                continue
            if not item.evidence_snippets:
                continue
            if self._evidence_snippet_quality(item.evidence_snippets[0]) < 0.35:
                continue
            strong.append(item)
            if len(strong) >= 2:
                break
        return strong

    def _filter_summary_points(
        self,
        items: list[EvidenceBackedPoint],
        kind: str,
    ) -> list[EvidenceBackedPoint]:
        kept: list[EvidenceBackedPoint] = []
        for item in items:
            text = (item.text or "").strip()
            if not text:
                continue
            if not item.evidence_chunk_ids or not item.evidence_snippets:
                continue
            if self._is_low_quality_summary_text(text, kind=kind):
                continue
            if kind == "follow_up" and item.confidence < 0.45:
                continue
            if kind == "discussion" and item.confidence < 0.25:
                continue
            kept.append(item)
        return kept[:8]

    @classmethod
    def _is_low_quality_summary_text(cls, text: str, kind: str) -> bool:
        if cls._is_low_quality_text(text, kind="decision"):
            return True
        lowered = cls._norm_text(text)
        words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
        if len(words) < 4:
            return True
        # Reject obvious ASR fragment/noise patterns frequently surfacing in summaries.
        noisy_phrases = (
            "speaker_",
            "warnings for next meetings",
            "suggested shades or whatever",
            "or whatever",
        )
        if any(p in lowered for p in noisy_phrases):
            return True
        if kind == "follow_up":
            # Require an explicit future/open-question cue for follow-up items.
            future_cues = (
                "next ",
                "follow up",
                "review ",
                "check ",
                "analyze ",
                "investigate ",
                "compare ",
                "evaluate ",
                "plan ",
                "later",
                "future",
                "before ",
                "open question",
                "question",
            )
            if not any(cue in lowered for cue in future_cues):
                if not text.strip().endswith("?"):
                    return True
        if kind == "discussion" and len(words) < 3:
            return True
        return False

    @classmethod
    def _looks_like_follow_up_candidate(cls, text: str) -> bool:
        lowered = cls._norm_text(text)
        cues = (
            "next ",
            "follow up",
            "review ",
            "check ",
            "analyze ",
            "investigate ",
            "compare ",
            "evaluate ",
            "before ",
            "question",
            "should",
            "need to",
        )
        return any(c in lowered for c in cues) or text.strip().endswith("?")

    @staticmethod
    def _evidence_snippet_quality(snippet: str) -> float:
        s = (snippet or "").strip()
        if not s:
            return 0.0
        words = re.findall(r"[A-Za-z']+", s)
        if not words:
            return 0.0
        alpha_ratio = sum((ch.isalpha() or ch.isspace()) for ch in s) / max(1, len(s))
        return round(min(1.0, len(words) / 20.0) * max(0.2, alpha_ratio), 3)

    @staticmethod
    def _extract_json_objects(text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        dec = json.JSONDecoder()
        for m in re.finditer(r"\{", text):
            try:
                obj, _ = dec.raw_decode(text[m.start() :])
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    @staticmethod
    def _normalize_summary_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
        summary = obj.get("summary")
        key_points = obj.get("key_points", [])
        discussion_points = obj.get("discussion_points", [])
        follow_up = obj.get("follow_up", obj.get("followups", []))
        if not isinstance(summary, str):
            return None
        if not isinstance(key_points, list):
            key_points = []
        if not isinstance(discussion_points, list):
            discussion_points = []
        if not isinstance(follow_up, list):
            follow_up = []

        # Recover when the model put a nested JSON dump in the summary field.
        nested = None
        s_stripped = summary.strip()
        if s_stripped.startswith("{") and '"summary"' in s_stripped:
            nested = LlamaCppBackend._parse_summary_json(s_stripped)
        if nested:
            summary = nested.get("summary", summary)
            if not key_points:
                key_points = nested.get("key_points", key_points)
            if not discussion_points:
                discussion_points = nested.get("discussion_points", discussion_points)
            if not follow_up:
                follow_up = nested.get("follow_up", follow_up)

        summary = LlamaCppBackend._clean_summary_text(summary)
        cleaned_points = []
        for x in key_points:
            pt = LlamaCppBackend._clean_summary_text(str(x))
            if not pt:
                continue
            # Drop obvious JSON-echo garbage in key points.
            if pt.startswith("{") and '"summary"' in pt:
                continue
            cleaned_points.append(pt)

        cleaned_discussion = LlamaCppBackend._clean_summary_list(discussion_points, limit=8)
        cleaned_follow_up = LlamaCppBackend._clean_summary_list(follow_up, limit=8)
        if not cleaned_discussion and cleaned_points:
            cleaned_discussion = cleaned_points[:]

        return {
            "summary": summary,
            "key_points": cleaned_points[:5],
            "discussion_points": cleaned_discussion,
            "follow_up": cleaned_follow_up,
        }

    @staticmethod
    def _clean_summary_text(s: str) -> str:
        text = (s or "").strip()
        if not text:
            return ""
        # Drop repeated prompt artifacts after a valid response.
        text = re.split(r"\bJSON:\b", text, maxsplit=1)[0].strip()
        # If the model echoed a JSON object as text, try to keep only the inner summary field content.
        if text.startswith("{") and '"summary"' in text:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
                    text = obj["summary"].strip()
            except Exception:
                pass
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _clean_summary_list(items: list[Any], limit: int) -> list[str]:
        out: list[str] = []
        for x in items or []:
            if isinstance(x, dict):
                candidate = x.get("text", x.get("point", x.get("item", "")))
            else:
                candidate = x
            pt = LlamaCppBackend._clean_summary_text(str(candidate))
            if not pt:
                continue
            if pt.startswith("{") and '"summary"' in pt:
                continue
            if pt in out:
                continue
            out.append(pt[:240])
            if len(out) >= limit:
                break
        return out

    def _best_evidence_chunks_for_point(
        self,
        point_text: str,
        chunks: list[dict[str, Any]],
    ) -> list[tuple[int, dict[str, Any]]]:
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            lexical_score = self._score_chunk_for_point(point_text, str(chunk.get("text", "") or ""))
            score = self._apply_asr_confidence_downrank(lexical_score, chunk)
            if score <= 0:
                continue
            scored.append((score, idx, chunk))
        if not scored:
            return []
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(score, chunk) for score, _, chunk in scored[:2]]

    @classmethod
    def _score_chunk_for_point(cls, point_text: str, chunk_text: str) -> int:
        lowered_point = (point_text or "").lower()
        lowered_chunk = (chunk_text or "").lower()
        if not lowered_chunk:
            return 0
        score = 0
        point_words = cls._keyword_set_from_text(lowered_point)
        chunk_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lowered_chunk))
        if point_words:
            score += min(len(point_words & chunk_words), 8)
        first_terms = [
            w.lower()
            for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", point_text or "")
            if w.lower() not in {"the", "and", "for", "with", "that", "this"}
        ]
        if first_terms and re.search(rf"\b{re.escape(first_terms[0])}\b", lowered_chunk):
            score += 2
        return score

    @staticmethod
    def _chunk_asr_confidence(chunk: dict[str, Any]) -> float:
        for key in ("asr_confidence", "avg_asr_confidence", "confidence"):
            try:
                if key in chunk and chunk.get(key) is not None:
                    return max(0.0, min(1.0, float(chunk.get(key))))
            except Exception:
                continue
        return 0.5

    def _apply_asr_confidence_downrank(self, lexical_score: int, chunk: dict[str, Any]) -> int:
        if lexical_score <= 0:
            return 0
        conf = self._chunk_asr_confidence(chunk)
        factor = 0.55 + (0.45 * conf)
        adjusted = int(round(lexical_score * factor))
        if adjusted <= 0 and lexical_score >= 3:
            return 1
        return adjusted

    @staticmethod
    def _score_summary_obj(obj: dict[str, Any]) -> int:
        summary = str(obj.get("summary", "") or "").strip()
        if not summary:
            return -1
        score = 0
        score += min(len(summary) // 40, 5)
        score += min(len(obj.get("key_points", [])), 5)
        score += min(len(obj.get("discussion_points", [])), 6)
        score += min(len(obj.get("follow_up", [])), 4)
        if "JSON:" not in summary:
            score += 2
        if not summary.startswith("{"):
            score += 2
        return score

    @staticmethod
    def _extract_chunk_prompt(meeting_id: str, chunk: dict[str, Any]) -> str:
        return (
            "You extract meeting decisions and action items from a transcript chunk.\n"
            "Return ONLY JSON with keys:\n"
            '- "decisions": array of {decision, confidence, uncertain}\n'
            '- "action_items": array of {action, owner, due_date, confidence, uncertain}\n'
            '- "flags": array of strings\n'
            "If none found, return empty arrays and optionally a flag.\n"
            "Keep text concise and factual. No markdown.\n"
            "Extraction rules:\n"
            "1) Action items must be clear, concrete, and verb-led (e.g., 'prepare cost estimate').\n"
            "2) Do NOT output fragments, single words, speaker labels, acknowledgements, or ASR-noisy text.\n"
            "3) If a candidate action/decision is unclear or incomplete, omit it and add a flag like 'insufficient_context'.\n"
            "4) Set owner only if explicitly stated in the chunk; otherwise owner must be null.\n"
            "5) Set due_date only if explicitly stated in the chunk; otherwise due_date must be null.\n"
            "6) Decisions should reflect explicit agreements/choices, not generic discussion topics.\n"
            "7) Prefer precision over recall: return fewer items rather than weak or speculative items.\n\n"
            f"Meeting ID: {meeting_id}\n"
            f"Chunk ID: {chunk.get('chunk_id')}\n"
            "Transcript chunk:\n"
            f"{chunk.get('text','')}\n\n"
            "JSON:"
        )

    @staticmethod
    def _parse_extraction_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        stripped = text.strip()
        candidates = [stripped]
        m = re.search(r"\{.*\}", stripped, flags=re.S)
        if m:
            candidates.append(m.group(0))
        for cand in candidates:
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            decisions = obj.get("decisions", [])
            actions = obj.get("action_items", obj.get("actions", []))
            flags = obj.get("flags", [])
            if not isinstance(decisions, list):
                decisions = []
            if not isinstance(actions, list):
                actions = []
            if not isinstance(flags, list):
                flags = []
            return {"decisions": decisions, "action_items": actions, "flags": flags}
        return None

    @staticmethod
    def _bounded_conf(v: Any, default: float) -> float:
        try:
            x = float(v)
        except Exception:
            x = default
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
        return round(x, 3)

    @staticmethod
    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def _dedupe_decisions(self, items: list[DecisionItem]) -> list[DecisionItem]:
        seen: dict[str, DecisionItem] = {}
        for it in items:
            k = self._norm_text(it.decision)
            if not k:
                continue
            if k not in seen:
                seen[k] = it
                continue
            prev = seen[k]
            prev.evidence_chunk_ids = sorted(set(prev.evidence_chunk_ids + it.evidence_chunk_ids))
            prev.evidence_snippets = self._merge_snippets(prev.evidence_snippets, it.evidence_snippets)
            prev.confidence = max(prev.confidence, it.confidence)
            prev.uncertain = prev.uncertain and it.uncertain
        return list(seen.values())[:30]

    def _dedupe_actions(self, items: list[ActionItem]) -> list[ActionItem]:
        seen: dict[str, ActionItem] = {}
        for it in items:
            k = self._norm_text(it.action)
            if not k:
                continue
            if k not in seen:
                seen[k] = it
                continue
            prev = seen[k]
            prev.evidence_chunk_ids = sorted(set(prev.evidence_chunk_ids + it.evidence_chunk_ids))
            prev.evidence_snippets = self._merge_snippets(prev.evidence_snippets, it.evidence_snippets)
            prev.confidence = max(prev.confidence, it.confidence)
            if not prev.owner and it.owner:
                prev.owner = it.owner
            if not prev.due_date and it.due_date:
                prev.due_date = it.due_date
            prev.uncertain = prev.uncertain and it.uncertain
        return list(seen.values())[:50]

    def _filter_decisions(self, items: list[DecisionItem]) -> tuple[list[DecisionItem], int]:
        kept: list[DecisionItem] = []
        dropped = 0
        for it in items:
            if self._is_low_quality_text(it.decision, kind="decision"):
                dropped += 1
                continue
            kept.append(it)
        return kept, dropped

    def _filter_actions(self, items: list[ActionItem]) -> tuple[list[ActionItem], int]:
        kept: list[ActionItem] = []
        dropped = 0
        for it in items:
            if self._is_low_quality_text(it.action, kind="action"):
                dropped += 1
                continue
            kept.append(it)
        return kept, dropped

    @classmethod
    def _is_low_quality_text(cls, text: str, kind: str) -> bool:
        s = (text or "").strip()
        if not s:
            return True
        lowered = s.lower()
        norm = cls._norm_text(s)
        words = [w for w in re.findall(r"[a-zA-Z']+", s) if w]
        if len(words) < 3:
            return True
        if len(norm) < 12:
            return True
        if re.fullmatch(r"speaker_\d+\s*:?", lowered):
            return True
        if re.match(r"^speaker_\d+\s*:\s*$", lowered):
            return True
        bad_exact = {
            "okay",
            "ok",
            "alright",
            "right",
            "so",
            "yeah",
            "yes",
            "no",
            "speaker_0:",
            "speaker_1:",
            "speaker_2:",
            "speaker_3:",
            "speaker_4:",
        }
        if lowered in bad_exact:
            return True
        # Reject obvious conversational fragments / acknowledgements often caused by ASR chunk boundaries.
        bad_prefixes = (
            "okay ",
            "ok ",
            "right ",
            "so ",
            "yeah ",
            "yes ",
            "no ",
        )
        if lowered.startswith(bad_prefixes) and len(words) < 6:
            return True
        # Action items should usually begin with a verb-like token, not a bare speaker label.
        if kind == "action":
            if lowered.startswith("speaker_") and len(words) < 6:
                return True
        return False

    @staticmethod
    def _merge_snippets(a: list[str], b: list[str]) -> list[str]:
        out: list[str] = []
        for s in list(a or []) + list(b or []):
            ss = str(s).strip()
            if not ss or ss in out:
                continue
            out.append(ss[:220])
            if len(out) >= 3:
                break
        return out

    @staticmethod
    def _evidence_snippet(chunk_text: str, extracted_text: str) -> str:
        text = re.sub(r"\s+", " ", (chunk_text or "").strip())
        if not text:
            return ""
        extracted_words = [w for w in re.findall(r"[A-Za-z']+", extracted_text or "") if w]
        if not extracted_words:
            return text[:220]
        # Try to anchor around the first extracted content word for a short audit-friendly snippet.
        anchor = extracted_words[0].lower()
        m = re.search(rf"\b{re.escape(anchor)}\b", text, flags=re.I)
        if not m:
            return text[:220]
        start = max(0, m.start() - 70)
        end = min(len(text), m.end() + 150)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet[:220]

    def _select_extraction_chunks(self, chunks: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
        if len(chunks) <= 6:
            return chunks
        summary_text = str(summary.get("summary", "") or "")
        key_points = [str(x) for x in summary.get("key_points", []) if str(x).strip()]
        discussion = []
        for x in summary.get("discussion_points", []) or []:
            if isinstance(x, dict):
                discussion.append(str(x.get("text", "")))
            else:
                discussion.append(str(x))
        follow_up = []
        for x in summary.get("follow_up", []) or []:
            if isinstance(x, dict):
                follow_up.append(str(x.get("text", "")))
            else:
                follow_up.append(str(x))
        cues = self._keyword_set_from_text(
            " ".join([summary_text] + key_points + discussion + follow_up)
        )
        # Bias toward decision/action language even if summary is generic.
        cues |= {
            "decide",
            "decided",
            "agree",
            "agreed",
            "action",
            "need",
            "will",
            "should",
            "next",
            "todo",
            "follow",
            "plan",
            "task",
        }
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            text = str(chunk.get("text", "") or "")
            score = self._score_chunk_for_extraction(text, cues)
            scored.append((score, idx, chunk))
        # Keep top scored chunks, but preserve chronology afterwards.
        scored.sort(key=lambda x: (-x[0], x[1]))
        keep_n = min(len(chunks), max(6, int(round(len(chunks) * 0.5))))
        chosen = scored[:keep_n]
        chosen.sort(key=lambda x: x[1])
        selected = [c for _, _, c in chosen]
        # If all scores are zero/near-zero, fall back to full extraction to avoid recall collapse.
        if not selected or max(s for s, _, _ in scored) <= 0:
            return chunks
        return selected

    @staticmethod
    def _keyword_set_from_text(text: str) -> set[str]:
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")]
        stop = {
            "speaker",
            "meeting",
            "discussion",
            "team",
            "project",
            "product",
            "design",
            "remote",
            "control",
            "controls",
            "their",
            "there",
            "about",
            "with",
            "from",
            "this",
            "that",
            "they",
            "have",
            "will",
            "should",
        }
        return {w for w in words if w not in stop}

    @staticmethod
    def _score_chunk_for_extraction(text: str, cues: set[str]) -> int:
        lowered = (text or "").lower()
        score = 0
        # Decision/action cue patterns.
        patterns = [
            r"\bwe should\b",
            r"\bwe need to\b",
            r"\bi will\b",
            r"\blet'?s\b",
            r"\bdecid(?:e|ed)\b",
            r"\bagree(?:d)?\b",
            r"\baction item\b",
            r"\bnext step\b",
            r"\bfollow up\b",
            r"\bcan you\b",
        ]
        for pat in patterns:
            if re.search(pat, lowered):
                score += 3
        # Reward overlap with summary-derived cues.
        if cues:
            chunk_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lowered))
            overlap = len(chunk_words & cues)
            score += min(overlap, 6)
        # Slightly prefer longer chunks (more context) but cap effect.
        score += min(len(re.findall(r"[A-Za-z']+", text or "")) // 40, 2)
        return score

    @staticmethod
    def _normalize_extraction_flags(flags: list[str], has_decisions: bool, has_actions: bool) -> list[str]:
        out: list[str] = []
        for f in flags:
            fs = str(f).strip()
            if not fs:
                continue
            lowered = fs.lower()
            if (has_decisions or has_actions) and (
                "no clear decisions or action items" in lowered
                or "no decisions or action items" in lowered
                or "no action items" in lowered
                or "no actions" in lowered
            ):
                continue
            if has_decisions and ("no clear decisions" in lowered or lowered == "no decisions"):
                continue
            out.append(fs[:120])
        return out
