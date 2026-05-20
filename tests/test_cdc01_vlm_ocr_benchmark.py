from scripts.benchmark_cdc01_vlm_ocr import PAGE_MARKERS, _mojibake, score_markers


def test_score_markers_accepts_clean_arabic_and_percent_ocr_artifact():
    text = "توجد غرامة التأخير بنسبة 1000/01 وتبلغ السقف 5 96."

    score = score_markers(text, PAGE_MARKERS[11])

    assert score["marker_hit_count"] == 3
    assert score["field_hits"]["penalties"] == ["غرامة التأخير", "1000/01", "5%"]


def test_score_markers_accepts_mojibake_aliases():
    text = f"{_mojibake('أمر بصرف')} خلال 30 يوما ثم الخلاص خلال 15 يوما."

    score = score_markers(text, PAGE_MARKERS[13])

    assert score["marker_hit_count"] == 3
    assert score["fields_with_hits"] == ["payment"]
