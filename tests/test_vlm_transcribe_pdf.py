from __future__ import annotations

from scripts.vlm_transcribe_pdf import (
    STRUCTURAL_TRANSCRIPTION_PROMPT,
    PageTranscription,
    build_openai_chat_payload,
    clean_markdown_response,
    format_transcription_output,
    parse_page_selection,
)


def test_structural_prompt_transcribes_without_answer_extraction():
    prompt = STRUCTURAL_TRANSCRIPTION_PROMPT

    assert "structured Markdown" in prompt
    assert "Preserve Arabic and French" in prompt
    assert "Do not translate" in prompt
    assert "Do not summarize" in prompt
    assert "answer procurement questions" in prompt
    assert "[ILLEGIBLE]" in prompt


def test_parse_page_selection_supports_ranges_bounds_and_all():
    assert parse_page_selection("all", page_count=8, max_pages=3) == [1, 2, 3]
    assert parse_page_selection("1,3-5,99", page_count=10, max_pages=8) == [1, 3, 4, 5]
    assert parse_page_selection("2-4,4,1", page_count=4) == [1, 2, 3, 4]


def test_openai_payload_contains_image_and_markdown_prompt():
    payload = build_openai_chat_payload(
        image_png=b"fake-png",
        prompt=STRUCTURAL_TRANSCRIPTION_PROMPT,
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=512,
        temperature=0,
    )

    content = payload["messages"][0]["content"]
    assert payload["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert payload["temperature"] == 0
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": STRUCTURAL_TRANSCRIPTION_PROMPT}


def test_clean_markdown_response_removes_code_fence_and_preserves_text():
    assert clean_markdown_response("```markdown\n# Title\n\n| A | B |\n```") == "# Title\n\n| A | B |"
    assert clean_markdown_response("") == "[ILLEGIBLE]"


def test_format_transcription_output_uses_page_markers():
    output = format_transcription_output(
        [
            PageTranscription(page=1, text="# Page one", seconds=1.2),
            PageTranscription(page=2, text="```markdown\n| Col |\n| --- |\n| x |\n```", seconds=2.3),
        ]
    )

    assert output == "[Page 1]\n# Page one\n\n[Page 2]\n| Col |\n| --- |\n| x |\n"
