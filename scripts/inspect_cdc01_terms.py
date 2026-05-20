from __future__ import annotations

from pathlib import Path


TERMS = [
    "صلوحية",
    "صلاحية",
    "120",
    "فتح العروض",
    "جلسة",
    "الضمان الوقتي",
    "الضمان النهائي",
    "غرام",
    "خطايا",
    "الت خير",
    "التأخير",
    "خلاص",
    "صرف",
    "الدفع",
    "أمر بصرف",
    "العرض الفني",
    "العرض المالي",
    "الوثائق",
    "بطاقة",
    "السجل الوطني",
    "CNSS",
    "جبائي",
    "الأثمان",
    "الأشمان",
]


def main() -> None:
    path = Path("/data/text_cache/CDC_01-2026.pdf.txt")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for term in TERMS:
        hits = [(i + 1, line.strip()) for i, line in enumerate(lines) if term in line]
        print(f"\n## {term}: {len(hits)}")
        for line_no, line in hits[:8]:
            print(f"{line_no}: {line[:240]}")


if __name__ == "__main__":
    main()
