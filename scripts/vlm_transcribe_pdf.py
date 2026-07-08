from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


STRUCTURAL_TRANSCRIPTION_PROMPT = """You are an absolute, zero-loss document transcription engine.

Analyze the provided page image and convert the entire visible page into structured Markdown.

Rules:
- Transcribe all visible text exactly as written.
- Preserve Arabic and French in the original language. Do not translate.
- Preserve all numbers, dates, percentages, amounts, punctuation, and units exactly.
- Preserve reading order: headers first, then body text, lists, forms, and tables.
- Reconstruct visible tables as Markdown tables, keeping row and column relationships.
- Reconstruct visible lists as Markdown lists.
- Do not summarize, classify, answer procurement questions, or extract only important facts.
- Do not infer missing words. If a word or region is unreadable, write [ILLEGIBLE] at that location.
- Output only Markdown. Do not wrap the answer in code fences.
"""


@dataclass(frozen=True)
class PageTranscription:
    page: int
    text: str
    seconds: float
    error: str | None = None


def pdf_page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as document:
        return int(document.page_count)


def render_pdf_page_png(pdf_path: Path, page_number: int, *, dpi: int) -> bytes:
    import fitz

    with fitz.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("png")


def parse_page_selection(value: str, *, page_count: int, max_pages: int | None = None) -> list[int]:
    value = str(value or "all").strip().lower()
    upper = page_count if max_pages is None else min(page_count, max_pages)
    if value in {"all", "auto"}:
        return list(range(1, upper + 1))

    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted({page for page in pages if 1 <= page <= upper})


def build_openai_chat_payload(
    *,
    image_png: bytes,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_png).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


def _extract_openai_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def call_openai_compatible(
    *,
    image_png: bytes,
    prompt: str,
    base_url: str,
    model: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    payload = build_openai_chat_payload(
        image_png=image_png,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = _post_json(f"{url}/chat/completions", payload, timeout=timeout)
    return _extract_openai_content(response)


def call_ollama_generate(
    *,
    image_png: bytes,
    prompt: str,
    base_url: str,
    model: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_png).decode("ascii")],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    response = _post_json(base_url.rstrip("/") + "/api/generate", payload, timeout=timeout)
    return str(response.get("response") or "")


def clean_markdown_response(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text or "[ILLEGIBLE]"


def format_transcription_output(results: list[PageTranscription]) -> str:
    sections: list[str] = []
    for result in results:
        text = clean_markdown_response(result.text)
        sections.append(f"[Page {result.page}]\n{text}")
    return "\n\n".join(sections).rstrip() + "\n"


def transcribe_pages(
    *,
    pdf_path: Path,
    pages: list[int],
    provider: str,
    base_url: str,
    model: str,
    dpi: int,
    timeout: float,
    max_tokens: int,
    temperature: float,
    prompt: str,
) -> list[PageTranscription]:
    results: list[PageTranscription] = []
    for page_number in pages:
        started = time.perf_counter()
        try:
            image_png = render_pdf_page_png(pdf_path, page_number, dpi=dpi)
            if provider == "ollama":
                text = call_ollama_generate(
                    image_png=image_png,
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                text = call_openai_compatible(
                    image_png=image_png,
                    prompt=prompt,
                    base_url=base_url,
                    model=model,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            results.append(
                PageTranscription(
                    page=page_number,
                    text=clean_markdown_response(text),
                    seconds=time.perf_counter() - started,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised against live VLM endpoints.
            results.append(
                PageTranscription(
                    page=page_number,
                    text="[ILLEGIBLE]",
                    seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def build_report(
    *,
    pdf_path: Path,
    provider: str,
    base_url: str,
    model: str,
    dpi: int,
    results: list[PageTranscription],
) -> dict[str, Any]:
    return {
        "schema": "vlm_transcription.v1",
        "document": pdf_path.name,
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "dpi": dpi,
        "pages": [
            {
                "page": result.page,
                "seconds": round(result.seconds, 3),
                "chars": len(result.text),
                "error": result.error,
            }
            for result in results
        ],
        "total_seconds": round(sum(result.seconds for result in results), 3),
        "errors": [result.error for result in results if result.error],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe PDF page images to structured Markdown with a VLM.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--pages", default="1")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--provider", choices=("openai", "ollama"), default=os.getenv("VLM_PROVIDER", "openai"))
    parser.add_argument("--base-url", default=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8003"))
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--dpi", type=int, default=int(os.getenv("VLM_DPI", "180")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("VLM_TIMEOUT_SECONDS", "240")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("VLM_MAX_OUTPUT_TOKENS", "4000")))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    page_count = pdf_page_count(args.pdf)
    pages = parse_page_selection(args.pages, page_count=page_count, max_pages=args.max_pages)
    if not pages:
        raise SystemExit("No valid pages selected.")

    results = transcribe_pages(
        pdf_path=args.pdf,
        pages=pages,
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        dpi=args.dpi,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        prompt=STRUCTURAL_TRANSCRIPTION_PROMPT,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_transcription_output(results), encoding="utf-8")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = build_report(
            pdf_path=args.pdf,
            provider=args.provider,
            base_url=args.base_url,
            model=args.model,
            dpi=args.dpi,
            results=results,
        )
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for result in results:
        status = "error" if result.error else "ok"
        print(f"page={result.page} status={status} chars={len(result.text)} seconds={result.seconds:.1f}")
        if result.error:
            print(f"  {result.error}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
