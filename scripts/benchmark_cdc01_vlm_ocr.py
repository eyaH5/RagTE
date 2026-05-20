from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


DEFAULT_PAGES = (8, 11, 13)


@dataclass(frozen=True)
class Marker:
    field: str
    text: str


PAGE_MARKERS: dict[int, tuple[Marker, ...]] = {
    8: (
        Marker("opening", "فتح العروض"),
        Marker("opening", "نفس اليوم"),
        Marker("opening", "جلسة واحدة"),
        Marker("caution", "الضمان الوقتي"),
        Marker("caution", "120 يوما"),
        Marker("information_sheet", "بطاقة الإرشادات"),
        Marker("rne", "السجل الوطني للمؤسسات"),
        Marker("administrative_documents", "وثيقة الضمان الوقتي"),
        Marker("administrative_documents", "تصريح على الشرف"),
        Marker("financial_documents", "التعهد المالي"),
        Marker("financial_documents", "جدول الأثمان"),
    ),
    11: (
        Marker("penalties", "غرامة التأخير"),
        Marker("penalties", "1000/01"),
        Marker("penalties", "5%"),
    ),
    13: (
        Marker("payment", "أمر بصرف"),
        Marker("payment", "30"),
        Marker("payment", "15"),
    ),
}


def _mojibake(text: str) -> str:
    try:
        return text.encode("utf-8").decode("latin-1")
    except UnicodeError:
        return text


def _marker_variants(marker: str) -> set[str]:
    variants = {marker, _mojibake(marker)}
    if "%" in marker:
        variants.add(marker.replace("%", " 96"))
        variants.add(marker.replace("%", "96"))
    return {variant for variant in variants if variant}


def score_markers(text: str, markers: tuple[Marker, ...]) -> dict[str, Any]:
    field_hits: dict[str, list[str]] = {}
    marker_hits: list[str] = []
    for marker in markers:
        if any(variant in text for variant in _marker_variants(marker.text)):
            marker_hits.append(marker.text)
            field_hits.setdefault(marker.field, []).append(marker.text)

    fields = sorted({marker.field for marker in markers})
    fields_with_hits = sorted(field_hits)
    return {
        "marker_hits": marker_hits,
        "marker_hit_count": len(marker_hits),
        "marker_total": len(markers),
        "field_hits": field_hits,
        "fields_with_hits": fields_with_hits,
        "field_hit_count": len(fields_with_hits),
        "field_total": len(fields),
    }


def render_page_png(pdf_path: Path, page_number: int, *, dpi: int) -> bytes:
    import fitz

    with fitz.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("png")


def extract_direct_text(pdf_path: Path, page_number: int) -> str:
    import fitz

    with fitz.open(pdf_path) as document:
        if page_number > document.page_count:
            return ""
        return document.load_page(page_number - 1).get_text("text") or ""


def extract_tesseract_text(image_png: bytes, *, lang: str, config: str) -> str:
    import pytesseract
    from PIL import Image

    image = Image.open(BytesIO(image_png))
    return pytesseract.image_to_string(image, lang=lang, config=config)


def extract_vlm_text(
    image_png: bytes,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    encoded = base64.b64encode(image_png).decode("ascii")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text from this Arabic/French procurement "
                            "document page. Preserve Arabic, numbers, dates, percentages, "
                            "and table structure. Output only the extracted text."
                        ),
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _summarize(label: str, page_result: dict[str, Any]) -> str:
    score = page_result["score"]
    return (
        f"{label:<12} chars={page_result['char_count']:<5} "
        f"markers={score['marker_hit_count']}/{score['marker_total']} "
        f"fields={score['field_hit_count']}/{score['field_total']} "
        f"hits={','.join(score['marker_hits']) or '-'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark current OCR versus an optional VLM on the CDC_01 pages "
            "that block Arabic extraction."
        )
    )
    parser.add_argument("--pdf", default="pdfs/CDC_01-2026.pdf")
    parser.add_argument("--pages", default=",".join(str(page) for page in DEFAULT_PAGES))
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--ocr-lang", default="ara+fra+eng")
    parser.add_argument("--ocr-config", default="--oem 1 --psm 6")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--vlm-base-url", default=os.getenv("VLM_BASE_URL", ""))
    parser.add_argument("--vlm-model", default=os.getenv("VLM_MODEL", ""))
    parser.add_argument("--vlm-api-key", default=os.getenv("VLM_API_KEY", "none"))
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument("--vlm-max-tokens", type=int, default=3000)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    pages = tuple(int(part.strip()) for part in args.pages.split(",") if part.strip())
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "pdf": str(pdf_path),
        "pages": {},
        "vlm_enabled": bool(args.vlm_base_url and args.vlm_model),
    }

    for page_number in pages:
        markers = PAGE_MARKERS.get(page_number, ())
        image_png = render_page_png(pdf_path, page_number, dpi=args.dpi)
        if output_dir:
            (output_dir / f"page_{page_number}.png").write_bytes(image_png)

        engines: dict[str, dict[str, Any]] = {}
        direct_text = extract_direct_text(pdf_path, page_number)
        engines["direct_pdf"] = {
            "text": direct_text,
            "char_count": len(direct_text),
            "score": score_markers(direct_text, markers),
        }

        tesseract_text = extract_tesseract_text(
            image_png,
            lang=args.ocr_lang,
            config=args.ocr_config,
        )
        engines["tesseract"] = {
            "text": tesseract_text,
            "char_count": len(tesseract_text),
            "score": score_markers(tesseract_text, markers),
        }

        if args.vlm_base_url and args.vlm_model:
            vlm_text = extract_vlm_text(
                image_png,
                base_url=args.vlm_base_url,
                model=args.vlm_model,
                api_key=args.vlm_api_key,
                timeout=args.vlm_timeout,
                max_tokens=args.vlm_max_tokens,
            )
            engines["vlm"] = {
                "text": vlm_text,
                "char_count": len(vlm_text),
                "score": score_markers(vlm_text, markers),
            }

        results["pages"][str(page_number)] = {
            "expected_fields": sorted({marker.field for marker in markers}),
            "expected_markers": [marker.text for marker in markers],
            "engines": engines,
        }

        print(f"\nPAGE {page_number}")
        for label, page_result in engines.items():
            print(_summarize(label, page_result))

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark JSON to {output}")


if __name__ == "__main__":
    main()
