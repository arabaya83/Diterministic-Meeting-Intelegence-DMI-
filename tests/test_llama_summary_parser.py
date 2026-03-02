from __future__ import annotations

import json

from ami_mom_pipeline.backends.llama_cpp_backend import LlamaCppBackend


def test_parse_summary_json_recovers_from_repeated_json_blocks() -> None:
    noisy = (
        '{\n'
        '  "summary": "The team discussed evaluation and decided to launch the product.",\n'
        '  "key_points": ["Decide to launch", "Budget met"]\n'
        '}\n'
        ' JSON: \n'
        '{\n'
        '  "summary": "The team discussed prototypes and budget issues. They agreed to present via camera.",\n'
        '  "key_points": ["Prototypes", "Budget issues", "Presentation via camera"]\n'
        '}\n'
    )
    parsed = LlamaCppBackend._parse_summary_json(noisy)
    assert parsed is not None
    assert "JSON:" not in parsed["summary"]
    assert parsed["summary"].startswith("The team discussed prototypes and budget issues")
    assert parsed["key_points"] == ["Prototypes", "Budget issues", "Presentation via camera"]


def test_parse_summary_json_recovers_nested_json_in_summary_field() -> None:
    nested = {
        "summary": (
            '{"summary":"Clean summary text.","key_points":["Point A","Point B"]} '
            "JSON: junk after valid payload"
        ),
        "key_points": [],
    }
    parsed = LlamaCppBackend._parse_summary_json(str(nested).replace("'", '"'))
    assert parsed is not None
    assert parsed["summary"] == "Clean summary text."
    assert parsed["key_points"] == ["Point A", "Point B"]


def test_parse_summary_json_preserves_reference_style_sections() -> None:
    payload = {
        "summary": "Fallback summary text.",
        "abstract": ["The team reviewed the interface design.", "They compared cost options."],
        "actions": ["The designer will prepare a prototype."],
        "decisions": ["The group chose a wheel interface."],
        "problems": ["Whether to include an LCD screen."],
        "key_points": ["Interface review"],
        "discussion_points": ["Interface design"],
        "follow_up": ["Prepare a prototype"],
    }
    parsed = LlamaCppBackend._parse_summary_json(json.dumps(payload))
    assert parsed is not None
    assert parsed["abstract"] == ["The team reviewed the interface design.", "They compared cost options."]
    assert parsed["actions"] == ["The designer will prepare a prototype."]
    assert parsed["decisions"] == ["The group chose a wheel interface."]
    assert parsed["problems"] == ["Whether to include an LCD screen."]
