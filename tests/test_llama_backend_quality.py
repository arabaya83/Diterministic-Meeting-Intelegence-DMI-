"""Quality-focused regression tests for Llama backend heuristics."""

from __future__ import annotations

from ami_mom_pipeline.backends.llama_cpp_backend import LlamaCppBackend
from ami_mom_pipeline.config import AppConfig
from ami_mom_pipeline.pipeline import stage_finalize_summary


def _backend() -> LlamaCppBackend:
    """Return a backend instance configured with default test settings."""
    return LlamaCppBackend(AppConfig())


def test_model_visible_chunk_text_strips_speaker_labels() -> None:
    """Prompt-visible chunk text should hide speaker labels."""
    chunk = {"text": "SPEAKER_1: We should prototype the remote.\nSPEAKER_2: I will compare costs."}
    cleaned = LlamaCppBackend._model_visible_chunk_text(chunk)
    assert "SPEAKER_1" not in cleaned
    assert "SPEAKER_2" not in cleaned
    assert "We should prototype the remote." in cleaned
    assert "I will compare costs." in cleaned


def test_select_extraction_chunks_keeps_explicit_cue_chunks_and_neighbors() -> None:
    """Chunk selection should keep explicit cue chunks and their neighbors."""
    backend = _backend()
    chunks = [
        {"chunk_id": "c1", "text": "SPEAKER_1: general brainstorming about shape and colors"},
        {"chunk_id": "c2", "text": "SPEAKER_2: general notes about the remote and interface"},
        {"chunk_id": "c3", "text": "SPEAKER_1: I will prepare the cost estimate by next week"},
        {"chunk_id": "c4", "text": "SPEAKER_3: broad discussion with no strong cues"},
        {"chunk_id": "c5", "text": "SPEAKER_2: some additional comments"},
        {"chunk_id": "c6", "text": "SPEAKER_1: more comments about design"},
        {"chunk_id": "c7", "text": "SPEAKER_4: closing remarks"},
    ]

    selected = backend._select_extraction_chunks(chunks, {"summary": "generic summary", "key_points": []})
    selected_ids = [chunk["chunk_id"] for chunk in selected]

    assert "c3" in selected_ids
    focus = next(chunk for chunk in selected if chunk["chunk_id"] == "c3")
    assert focus["_prev_chunk_id"] == "c2"
    assert focus["_next_chunk_id"] == "c4"


def test_extract_chunk_prompt_uses_window_context_without_speaker_labels() -> None:
    """Extraction prompts should include surrounding context without speaker labels."""
    prompt = LlamaCppBackend._extract_chunk_prompt(
        "IS1001c",
        {"chunk_id": "c2", "text": "SPEAKER_2: I will prepare the estimate."},
        prev_chunk={"chunk_id": "c1", "text": "SPEAKER_1: We need a cost comparison."},
        next_chunk={"chunk_id": "c3", "text": "SPEAKER_3: Next week works for me."},
    )

    assert "Previous context chunk:" in prompt
    assert "Focus chunk:" in prompt
    assert "Next context chunk:" in prompt
    assert "SPEAKER_" not in prompt
    assert "I will prepare the estimate." in prompt
    assert "We need a cost comparison." in prompt
    assert "Next week works for me." in prompt


def test_evidence_snippet_strips_speaker_labels() -> None:
    """Evidence snippets should remove speaker-label noise."""
    snippet = LlamaCppBackend._evidence_snippet(
        "SPEAKER_1: We should build a prototype next week. SPEAKER_2: Agreed.",
        "build a prototype",
    )
    assert "SPEAKER_" not in snippet
    assert "build a prototype" in snippet.lower()
    assert len(snippet) <= 160


def test_stage_finalize_summary_adds_extraction_actions_to_follow_up(tmp_path) -> None:
    """Finalize-summary should carry extraction actions into follow-up items."""
    cfg = AppConfig()
    paths = tmp_path / "meeting"
    paths.mkdir()
    summary_out = {
        "meeting_id": "IS1001c",
        "summary": "The team discussed prototype options.",
        "key_points": ["Prototype options"],
        "discussion_points": [],
        "follow_up": [],
        "prompt_template_version": "llama-cpp-v1",
        "backend": "llama_cpp",
    }
    extract_out = {
        "meeting_id": "IS1001c",
        "decisions": [],
        "action_items": [
            {
                "action": "Prepare a prototype for the next meeting",
                "owner": None,
                "due_date": None,
                "evidence_chunk_ids": ["IS1001c_chunk_0003"],
                "evidence_snippets": ["Prepare a prototype before the next meeting."],
                "confidence": 0.72,
                "uncertain": False,
            }
        ],
        "flags": [],
    }

    class _Paths:
        """Minimal stand-in exposing the artifact directory expected by the stage."""

        meeting_artifacts_dir = paths

    finalized = stage_finalize_summary(cfg, _Paths(), "IS1001c", summary_out, extract_out)
    assert finalized["follow_up"]
    assert finalized["follow_up"][0]["text"] == "Prepare a prototype for the next meeting"


def test_speculative_action_text_is_rejected() -> None:
    """Speculative brainstorming fragments should not count as actions."""
    backend = _backend()
    assert backend._is_speculative_action_text("Find a spongy fruit")
    assert backend._is_speculative_action_text("Control TV with a fruit")
    assert not backend._is_speculative_action_text("Prepare a prototype before the next meeting")


def test_follow_up_filter_rejects_brainstorm_artifacts() -> None:
    """Follow-up filtering should reject brainstorming artifacts from noisy output."""
    backend = _backend()
    assert backend._is_low_quality_summary_text("Find a spongy fruit", kind="follow_up")
    assert backend._is_low_quality_summary_text("Control TV with a fruit", kind="follow_up")
    assert not backend._is_low_quality_summary_text(
        "Define the next task to be done before the next meeting",
        kind="follow_up",
    )


def test_follow_up_normalization_rewrites_transcript_fragments() -> None:
    """Follow-up normalization should rewrite repeated transcript fragments."""
    backend = _backend()
    assert (
        backend._normalize_summary_point_text(
            "show various investigation done during preview",
            kind="follow_up",
        )
        == "Review prior investigation findings"
    )
    assert (
        backend._normalize_summary_point_text(
            "define the next task to be done before the next meeting",
            kind="follow_up",
        )
        == "Define tasks before the next meeting"
    )


def test_key_point_filter_drops_brainstorm_artifacts() -> None:
    """Key-point filtering should discard low-quality brainstorm artifacts."""
    backend = _backend()
    filtered = backend._filter_key_points(
        [
            "Banana-shaped remote control",
            "Spongy material for the remote's surface",
            "Lighted buttons for better visibility",
            "Symmetrical design for left-handed use",
        ]
    )
    assert filtered == [
        "Lighted buttons for better visibility",
        "Symmetrical design for left-handed use",
    ]


def test_summary_narrative_downgrades_unverified_agreement_language() -> None:
    """Narrative normalization should avoid unsupported agreement claims."""
    backend = _backend()
    chunks = [
        {"chunk_id": "c1", "text": "SPEAKER_1: We should review prior research next time."},
        {"chunk_id": "c2", "text": "SPEAKER_2: We need to define tasks before the next meeting."},
    ]
    normalized = backend._normalize_summary_narrative(
        "The team agreed to finalize the design in the next meeting and prepare a prototype.",
        chunks,
    )
    assert "agreed to" not in normalized.lower()
    assert "discussed plans to" in normalized.lower()


def test_coverage_chunks_for_prompt_spans_late_meeting_content() -> None:
    """Prompt coverage should include early, middle, and late chunks."""
    backend = _backend()
    chunks = [
        {"chunk_id": f"c{i:02d}", "text": f"SPEAKER_1: generic content section {i}"}
        for i in range(1, 41)
    ]
    coverage = backend._coverage_chunks_for_prompt(chunks, limit=12)
    coverage_ids = [chunk["chunk_id"] for chunk in coverage]

    assert "c01" in coverage_ids
    assert "c40" in coverage_ids
    assert any(chunk_id in coverage_ids for chunk_id in {"c20", "c21", "c22"})
