from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any
from urllib import request


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def render_pdf_page_png(pdf_path: Path, page_number: int, dpi: int) -> bytes:
    import fitz

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number - 1)
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("png")


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    url = args.base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    text_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Return exactly this JSON and nothing else: {\"answer\":\"hello\"}",
            }
        ],
        "temperature": 0,
        "max_tokens": 80,
    }
    text_response = post_json(f"{url}/chat/completions", text_payload, args.timeout)
    print("TEXT_CONTENT_START")
    print(extract_content(text_response))
    print("TEXT_CONTENT_END")

    if args.pdf:
        image_png = render_pdf_page_png(args.pdf, args.page, args.dpi)
        image_b64 = base64.b64encode(image_png).decode("ascii")
        image_payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this page. Return strict JSON only: {\"visible_title\":\"...\"}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        image_response = post_json(f"{url}/chat/completions", image_payload, args.timeout)
        print("IMAGE_CONTENT_START")
        print(extract_content(image_response))
        print("IMAGE_CONTENT_END")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
