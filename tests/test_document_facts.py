from __future__ import annotations

from pathlib import Path

import pytest

import ingest
from api.services.rag import (
    HALLUCINATION_SIGNALS,
    _answer_from_mined_facts,
    _detect_answer_language,
    _fact_list_field_for_question,
    _language_instruction,
    answer_from_document_facts,
    build_tender_checklist_answer,
)
from ingest import _entries_need_arabic_ocr, _should_use_direct_pdf_text, extract_document_facts


def _facts_from_text(filename: str, text: str, section: str = "general") -> dict:
    return extract_document_facts(
        [text],
        [{"source": filename, "page": "1", "section": section, "chunk_index": 0}],
    )


def test_text_quality_metadata_detects_page_gaps():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "Objet du marche et soumission."},
            {"page": "2", "text": "Offre technique."},
            {"page": "5", "text": "Modalites de paiement."},
        ],
        page_count=5,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "partial_pages"
    assert quality["page_gap_count"] == 2
    assert quality["missing_page_ranges"] == ["3-4"]
    assert quality["text_source"] == "docling_ocr"


def test_text_quality_metadata_prioritizes_arabic_noisy_over_page_gaps():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "طلب عروض لاقتناء مواد إعلامية. " * 20},
            {"page": "4", "text": "العرض الفني والضمان الوقتي. " * 20},
        ],
        page_count=4,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "arabic_noisy"
    assert quality["page_gap_count"] == 2
    assert quality["missing_page_ranges"] == ["2-3"]


def test_text_quality_metadata_keeps_french_page_gaps_as_partial_pages():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "Objet du marche et soumission."},
            {"page": "4", "text": "Modalites de paiement."},
        ],
        page_count=4,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "partial_pages"
    assert quality["page_gap_count"] == 2


def test_text_quality_metadata_detects_noisy_non_arabic_ocr_before_page_gaps():
    quality = ingest._build_text_quality_metadata(
        [
            {"page": "1", "text": "@@@ ### !!! ???"},
            {"page": "4", "text": "$$$ *** !!! ???"},
        ],
        page_count=4,
        text_source="docling_ocr",
    )

    assert quality["mode"] == "noisy_ocr"
    assert quality["page_gap_count"] == 2


def test_extract_and_chunk_prefers_tsb_pdf_text_layer_when_cache_has_page_gaps(monkeypatch):
    pdf_path = Path(__file__).parents[1] / "pdfs" / "TUNISIAN SAUDI BANK.pdf"
    if not pdf_path.exists():
        pytest.skip("TSB fixture PDF is not available")

    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    chunks, metas, _ids = ingest.extract_and_chunk(str(pdf_path), pdf_path.name)

    assert chunks
    assert metas[0]["text_quality"]["text_source"] == "pdf_text_layer"
    assert metas[0]["text_quality"]["preferred_source"] == "pdf_text_layer"

    joined = "\n".join(chunks).lower()
    assert "voie postale" in joined
    assert "24 février 2025" in joined
    assert "cnss" in joined
    assert "rne" in joined


def test_extract_and_chunk_supports_plain_text_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    path = tmp_path / "consultation.txt"
    path.write_text(
        "La presente consultation a pour objet l'acquisition de consommables informatiques.\n"
        "La date limite de reception des offres est fixee au 10/06/2025.",
        encoding="utf-8",
    )

    chunks, metas, ids = ingest.extract_and_chunk(str(path), path.name)

    assert chunks
    assert metas
    assert ids
    assert metas[0]["source"] == path.name
    facts = extract_document_facts(chunks, metas)
    assert facts["subject"]["text"].startswith("l'acquisition de consommables")
    assert facts["deadline"]["text"] == "10/06/2025"


def test_extract_and_chunk_supports_csv_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    path = tmp_path / "items.csv"
    path.write_text("Designation,Quantite\nToner HP,20\nScanner,2\n", encoding="utf-8")

    chunks, metas, _ = ingest.extract_and_chunk(str(path), path.name)

    joined = "\n".join(chunks)
    assert "Designation | Quantite" in joined
    assert "Toner HP | 20" in joined
    assert metas[0]["source"] == path.name


def test_extract_and_chunk_supports_docx_files(monkeypatch, tmp_path):
    docx = pytest.importorskip("docx")
    monkeypatch.setattr(ingest, "_write_text_cache", lambda *args, **kwargs: None)

    path = tmp_path / "consultation.docx"
    document = docx.Document()
    document.add_paragraph("La presente consultation a pour objet la fourniture de licences.")
    document.add_heading("Pieces administratives", level=1)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Document"
    table.cell(0, 1).text = "Exigence"
    table.cell(1, 0).text = "CNSS"
    table.cell(1, 1).text = "Attestation d'affiliation"
    document.save(path)

    chunks, metas, _ = ingest.extract_and_chunk(str(path), path.name)

    joined = "\n".join(chunks)
    assert "fourniture de licences" in joined
    assert "CNSS | Attestation d'affiliation" in joined
    assert metas[0]["source"] == path.name
    assert any(meta.get("doc_type") == "docx" for meta in metas)
    assert any(meta.get("location") == "Section: Pieces administratives" for meta in metas)
    assert any(meta.get("section_heading") == "Pieces administratives" for meta in metas)


def test_direct_pdf_text_checker_rejects_tender_without_deadline_value():
    entries = [
        {
            "page": "1",
            "text": """
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES.
            Les offres demeureront valables pendant 90 jours apres la date limite de reception des offres.
            L'offre technique et l'offre financiere doivent etre placees dans deux enveloppes separees.
            Ces enveloppes, en plus de la caution provisoire et du dossier administratif, seront deposees.
            Conditions de paiement et penalite de retard sont precisees dans le cahier des charges.
            """,
        }
    ]

    assert _should_use_direct_pdf_text(entries, page_count=1) is False


def test_direct_pdf_text_checker_accepts_tender_with_deadline_value():
    entries = [
        {
            "page": "1",
            "text": """
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES.
            Les offres doivent parvenir par voie postale.
            La date limite de reception des offres est fixee au 24 fevrier 2025.
            L'offre technique et l'offre financiere doivent etre placees dans deux enveloppes separees.
            Conditions de paiement et penalite de retard sont precisees dans le cahier des charges.
            """,
        }
    ]

    assert _should_use_direct_pdf_text(entries, page_count=1) is True


def test_direct_pdf_text_checker_rejects_sparse_large_scanned_document():
    entries = [
        {
            "page": "1",
            "text": """
            REPUBLIQUE TUNISIENNE
            APPEL D'OFFRES N 01/2025 CAHIER DES CHARGES
            Quelques fragments extraits seulement.
            """,
        },
        {"page": "27", "text": "Date limite 20/05/2025."},
        {"page": "50", "text": "Prix Qte."},
    ]

    assert _should_use_direct_pdf_text(entries, page_count=50) is False


def test_extract_document_facts_marks_unusable_ocr_scan():
    chunks = [
        "ZO/S\nEST\nSEE\nIAE\nIIS\nFES\nS/E\nEEE\nSSS\nOO",
        "TEE\nESE\nSOO\nAI\nII\n12\n05\n3032\nNO\nNN",
        "AE\nRE\nSE\nTT\nLL\nPO\nQO\nEE\nSS\n00",
        "II\nII\nIII\nEE\nAA\nRR\nTT\nOO\nSS\nUU",
        "MM\nNN\nPP\nQQ\nRR\nSS\nTT\nVV\nWW\nXX",
    ]
    metas = [
        {"source": "UNKNOWN.pdf", "page": str(index + 1), "section": "general", "chunk_index": index}
        for index in range(len(chunks))
    ]

    facts = extract_document_facts(chunks, metas)

    assert "extraction_warning" in facts
    assert facts["extraction_warning"]["level"] == "warning"
    assert "Qualité d'extraction insuffisante" in facts["extraction_warning"]["text"]


def test_arabic_ocr_checker_retries_sparse_large_document_after_docling():
    entries = [
        {"page": "1", "text": "REPUBLIQUE TUNISIENNE APPEL D'OFFRES quelques fragments."},
        {"page": "25", "text": "TUNEPS tableau incomplet."},
        {"page": "50", "text": "Prix Qte."},
    ]

    assert _entries_need_arabic_ocr(entries) is True


def test_arabic_weak_profile_ocr_reinforcement_keeps_clause_pages():
    entries = [
        {"page": str(page), "text": "نص عربي عام من كراس الشروط."}
        for page in range(1, 23)
    ]
    entries[6]["text"] = (
        "إرسال العرض الفني والعرض المالي على منظومق الشراء العموميه علو الخط. "
        "وثيقة الضمان الوقتي وبطاقة الإرشادات والسجل الوطني."
    )
    entries[8]["text"] = "مدة الضمان من تاريخ القبول الوقتي."
    entries[10]["text"] = "غرامق الت خير وخطايا التأخير."
    entries[12]["text"] = "خلاص صاحب الصفقة وأمر بصرف المبالغ وفاتورة."

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {"tender_profile": {"coverage": {"core_ratio": 0.25}}},
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages
    assert len(pages) <= ingest.OCR_REINFORCE_ARABIC_MAX_PAGES


def test_arabic_weak_profile_ocr_reinforcement_detects_clean_arabic_pages():
    entries = [
        {"page": str(page), "text": "نص عربي عام من كراس الشروط."}
        for page in range(1, 23)
    ]
    entries[7]["text"] = (
        "فتح العروض في نفس اليوم جلسة واحدة. "
        "وثيقة الضمان الوقتي وبطاقة الإرشادات والسجل الوطني للمؤسسات. "
        "التعهد المالي وجدول الأثمان."
    )
    entries[10]["text"] = "غرامة التأخير بنسبة 1000/01 في اليوم وبسقف 5%."
    entries[12]["text"] = "يصدر المشتري العمومي أمر بصرف المبالغ في أجل 30 يوما ثم الخلاص في أجل 15 يوما."

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {"tender_profile": {"coverage": {"core_ratio": 0.25}}},
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages


def test_arabic_weak_profile_ocr_reinforcement_uses_preview_facts():
    entries = [
        {"page": str(page), "text": "latin OCR garbage without useful markers"}
        for page in range(1, 23)
    ]

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {
            "subject": {"text": "طلب عروض لاقتناء مواد إعلامية"},
            "submission_method": {"text": "منظومة الشراء العمومي على الخط تونابس"},
            "tender_profile": {"coverage": {"core_ratio": 0.25}},
        },
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages


def test_weak_long_profile_ocr_reinforcement_keeps_front_clause_pages():
    entries = [
        {"page": str(page), "text": "fragment OCR de cahier des charges"}
        for page in range(1, 23)
    ]

    pages = ingest._target_pages_for_ocr_reinforcement(
        entries,
        {"tender_profile": {"coverage": {"core_ratio": 0.25}}},
    )

    assert pages[:14] == list(range(1, 15))
    assert 8 in pages
    assert 11 in pages
    assert 13 in pages


def test_extract_and_chunk_reinforces_weak_core_facts_with_targeted_ocr(monkeypatch):
    direct_entries = [
        {
            "page": "1",
            "text": "CAHIER DES CHARGES. Les offres sont deposees via TUNEPS. Validite indiquee au DPC.",
        },
        {
            "page": "23",
            "text": "Pieces objet de changement dans le cadre du marche: 1070119891Developpingunit, = 1",
        },
    ]
    seen = {}

    monkeypatch.setattr(ingest, "_extract_text_entries_pypdf", lambda path: (direct_entries, 23))
    monkeypatch.setattr(ingest, "_should_use_direct_pdf_text", lambda entries, page_count: True)
    monkeypatch.setattr(ingest, "_entries_need_arabic_ocr", lambda entries: False)
    monkeypatch.setattr(ingest, "_extracted_facts_need_ocr_reinforcement", lambda facts, entries: True)
    monkeypatch.setattr(ingest, "_target_pages_for_ocr_reinforcement", lambda entries, facts: [1, 2, 3])
    monkeypatch.setattr(ingest, "_write_text_cache", lambda filename, entries: None)

    def fake_tesseract(path, filename, pages=None):
        seen["pages"] = pages
        return [
            {
                "page": "1",
                "text": "Objet : Acquisition de pieces de rechange pour imprimantes. Date limite : 20/05/2026.",
            }
        ]

    monkeypatch.setattr(ingest, "_extract_text_entries_tesseract", fake_tesseract)

    chunks, metas, ids_out = ingest.extract_and_chunk("demo.pdf", "demo.pdf")

    assert seen["pages"] == [1, 2, 3]
    assert any("Acquisition de pieces de rechange" in chunk for chunk in chunks)
    assert len(chunks) == len(metas) == len(ids_out)


def test_requested_items_are_extracted_from_short_license_request():
    facts = _facts_from_text(
        "BH_ASSURANCE.pdf",
        """
        BH ASSURANCE CONSULTATION-DSI-05-2025 Objet : Consultation pour le renouvellement
        des Licences Veeam. Rubriques Quantite Id de Licence Support Id
        Veeam Backup for Microsoft 365 125 utilisateurs 99D1B297 #02714153
        Veeam Availability Suite 100 Instances E98D6984 #02714151
        """,
    )

    requested = facts["requested_items"]
    normalized = requested["text"].lower()
    assert "veeam backup for microsoft 365 : 125 utilisateurs" in normalized
    assert "id licence 99d1b297" in normalized
    assert "support id #02714153" in normalized
    assert "veeam availability suite : 100 instances" in normalized
    assert "support id #02714151" in normalized
    assert _fact_list_field_for_question("Quel est le support id ?") == "requested_items"
    assert _fact_list_field_for_question("Quelles licences sont demandées ?") == "requested_items"


def test_requested_items_are_extracted_from_multiline_license_table():
    facts = _facts_from_text(
        "BH_ASSURANCE.pdf",
        """
        BH ASSURANCE
        CONSULTATION-DSI-05-2025
        Objet : Consultation pour le renouvellement des Licences Veeam.
        Rubriques Quantite Id de Licence Support Id
        Veeam Backup for
        Microsoft 365
        125
        utilisateurs
        99D1B297-263D-B66F-3705-
        C50657F317E0
        #02714153
        Veeam Availability
        Suite
        100 Instances E98D6984-C8E3-B99A-
        2D9D-9C501F16DBF3
        #02714151
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "veeam backup for microsoft 365 : 125 utilisateurs" in normalized
    assert "support id #02714153" in normalized
    assert "veeam availability suite : 100 instances" in normalized
    assert "support id #02714151" in normalized


def test_requested_items_and_deadline_are_extracted_from_quantity_first_table():
    facts = _facts_from_text(
        "BANQUE_CENTRALE_DE_TUNISIE.pdf",
        """
        Quantite Designation
        20 TONER LEXMARK CX 431ADW (20NOX20) : BLEU
        20 TONER LEXMARK CX 431 ADW (20NOX10) : NOIR
        20 BOUTEILLE RECUP TONER (WASTE TONER BOTTLE) LEXMARK CX 431ADW (20NOW00)
        Votre offre doit etre expediee par voie postale ou deposee directement au Bureau d'Ordre
        Central au plus tard le 27/01/2025, sous plis ferme et anonyme.
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "toner lexmark cx 431adw" in normalized
    assert ": 20" in normalized
    assert "bouteille" in normalized
    assert facts["deadline"]["text"] == "27/01/2025"


def test_requested_items_are_extracted_from_generic_product_table():
    facts = _facts_from_text(
        "GENERIC_TABLE.pdf",
        """
        Rubriques Quantite Reference Support ID
        Microsoft 365 Business Premium 50 licences MS365-BP-50 #112233
        Fortinet FortiGate 100F 2 unites FG100F-BDL #445566
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "microsoft 365 business premium : 50 licences" in normalized
    assert "support id #112233" in normalized
    assert "fortinet fortigate 100f : 2 unites" in normalized
    assert "support id #445566" in normalized


def test_requested_items_are_extracted_from_item_designation_quantity_table():
    facts = _facts_from_text(
        "GENERIC_BORDEREAU.pdf",
        """
        Bordereau des prix
        Item Designation Unite Qte
        1 Firewall FortiGate 100F Unite 2
        2 Switch 24 ports PoE Unite 4
        3 Licence antivirus poste 36 mois Licences 150
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "firewall fortigate 100f : 2 unite" in normalized
    assert "switch 24 ports poe : 4 unite" in normalized
    assert "licence antivirus poste 36 mois : 150 licences" in normalized

    answer, _ = _answer_from_mined_facts(
        "GENERIC_BORDEREAU.pdf",
        "Quels articles sont demandés ?",
        facts,
    )
    assert "Firewall FortiGate 100F" in answer
    assert "Switch 24 ports PoE" in answer


def test_requested_items_ignore_metric_and_annex_table_noise():
    facts = _facts_from_text(
        "GENERIC_BORDEREAU.pdf",
        """
        Bordereau des prix
        Item Designation Unite Qte
        1 et versions ulterieures Windows Server 2016 Unite 11
        2 Admin externes / Unite 35
        3 de ressources a proteger Unite 25
        4 DC 02 ANNEXE 3 Modele de bordereau des prix LOT 2
        5 Mise en Place d'une Solution PAM Unite 1
        """,
    )

    normalized = facts["requested_items"]["text"].lower()
    assert "mise en place d'une solution pam : 1 unite" in normalized
    assert "windows server" not in normalized
    assert "admin externes" not in normalized
    assert "ressources a proteger" not in normalized
    assert "modele de bordereau" not in normalized


def test_rose_blanche_financial_documents_and_subject_are_extracted():
    facts = _facts_from_text(
        "Rose_Blanche.pdf",
        """
        ARTICLE 1 - OBJET DE L'APPEL D'OFFRES
        Le présent appel d'offres a pour objet la fourniture, l'installation, la configuration,
        l'intégration et la mise en production de solutions de cybersécurité répondant aux
        besoins de la rose Blanche selon la répartition suivante :
        1. LOT 1 Solution de type Antivirus + EDR (EndPoint Detection and Response)
        2. Lot 2 : Mise en place d'une Solution PAM

        3. Fichier Zip INTERIEURE « F » OFFRE FINANCIERE :
        N° DOCUMENTS OPERATION A REALISER AUTHENTIFICATION
        F1 La soumission Remplir le modèle fourni en ANNEXE 2 Original du document remis
        par Rose Blanche dûment complété par le soumissionnaire Datée et portant signature
        et cachet du soumissionnaire à la fin du document.
        F2 Le bordereau des prix (Remplir le modèle fourni en annexe) Original du document
        remis par Rose Blanche dûment complété par le soumissionnaire Paraphe, signature
        & cachet du soumissionnaire.
        """,
    )

    assert "Antivirus + EDR" in facts["subject"]["text"]
    assert "PAM" in facts["subject"]["text"]
    normalized_financial = facts["financial_documents"]["text"].lower()
    assert "la soumission" in normalized_financial
    assert "bordereau des prix" in normalized_financial


def _rose_blanche_table_facts() -> dict:
    chunks = [
        """
        ARTICLE 1 - OBJET DE L'APPEL D'OFFRES
        Le present appel d'offres a pour objet la fourniture, l'installation, la configuration,
        l'integration et la mise en production de solutions de cybersecurite repondant aux
        besoins de la rose Blanche selon la repartition suivante :
        1. LOT 1 Solution de type Antivirus + EDR (EndPoint Detection and Response)
        2. Lot 2 : Mise en place d'une Solution PAM
        """,
        """
        LOT 1 : Solution de type Antivirus + EDR
        Nombre de Endpoint a proteger :
        PC : 1300
        Serveur : 120
        Mobile : 100
        """,
        """
        LOT 2 : Mise en place d'une Solution PAM
        Nombre des utilisateurs de la plateforme 35 Admin externes / 25 Admin internes
        Nombre de ressources a proteger 150
        """,
        """
        ANNEXE 3 Modele de bordereau des prix (Lot 1)
        LOT 1: Une solution Antivirus + EDR
        Item Designation Unite Qte Prix U. HT Total HT
        1 Mise en Place dune Solution Antivirus + EDR Unite 1
        4 Installation et mise en place de la solution Unite 1
        5 Formation et Transfert de competences Unite 1
        6 Support sur 36 mois a payer annuellement sur Trois ans Unite 1
        """,
        """
        ANNEXE 3 Modele de bordereau des prix (Lot 2)
        LOT 2: Une solution PAM
        Item Designation Unite Qte Prix U. HT Total HT
        1 Mise en Place dune Solution PAM selon le sizing propose Unite 1
        4 Installation et mise en place de la solution Unite 1
        5 Formation et Transfert de competences Unite 1
        6 Support sur 36 mois a payer annuellement sur Trois ans Unite 1
        """,
    ]
    metas = [
        {"source": "Rose_Blanche.pdf", "page": str(page), "section": "general", "chunk_index": page - 1}
        for page in (2, 12, 31, 39, 40)
    ]
    return extract_document_facts(chunks, metas)


def test_rose_blanche_mined_facts_cover_lots_metrics_and_bordereaux():
    facts = _rose_blanche_table_facts()

    mined = facts["mined_facts"]
    normalized = mined["text"].lower()
    assert "lot 1" in normalized
    assert "antivirus" in normalized
    assert "lot 2" in normalized
    assert "pam" in normalized
    assert "nombre de endpoint pc : 1300" in normalized
    assert "nombre de endpoint serveur : 120" in normalized
    assert "nombre de endpoint mobile : 100" in normalized
    assert "35 admin externes" in normalized
    assert "ressources" in normalized
    assert "150" in normalized
    assert "support sur 36 mois" in normalized


def test_mined_facts_answer_table_questions_without_generic_rag():
    facts = _rose_blanche_table_facts()

    answer, _ = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Quels sont les lots demandes ?",
        facts,
    )
    assert "Lot 1" in answer
    assert "Antivirus + EDR" in answer
    assert "Lot 2" in answer
    assert "PAM" in answer

    answer, metas = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Combien de endpoints sont demandes pour le lot Antivirus + EDR ?",
        facts,
    )
    assert "1300" in answer
    assert "120" in answer
    assert "100" in answer
    assert metas[0]["source"] == "Rose_Blanche.pdf"

    answer, _ = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Combien d'administrateurs sont prevus pour la solution PAM ?",
        facts,
    )
    assert "35 Admin externes" in answer
    assert "25 Admin internes" in answer

    answer, _ = _answer_from_mined_facts(
        "Rose_Blanche.pdf",
        "Quels sont les elements du bordereau des prix pour le lot 2 ?",
        facts,
    )
    assert "Mise en Place dune Solution PAM" in answer
    assert "Support sur 36 mois" in answer


def _republique_tunisienne_table_facts() -> dict:
    chunks = [
        """
        OFFICE DE LA TOPOGRAPHIE ET DU CADASTRE
        إستشارة عدد 2025/17 لإقتناء 04 Coupeuses de plans A0
        يتم تقديم العروض عبر منظومة الشراء العمومي على الخط TUNEPS على الموقع www.tuneps.tn
        لقبول العروض يوم 2025/04/10
        """,
        """
        Caractéristiques Techniques Coupeuse de Plan A0 (36 ")
        Caractéristique technique Minimum demandé
        Fonctionnalité professionnelle grand format papier A0
        Type de coupe Manuelle
        Coupe bidirectionnel oui
        Longueur de coupe : Papier A0 | > 1190 mm
        Orientation papier A0 à découper | Portrait et paysage
        Graduations mm
        Dispositif de pression Automatique
        Table de coupe Métallique
        Bac de récupération des chutes papiers Avec bac de récupération des chutes papiers
        Equipement de sécurité Tête de coupe carénée
        Protection intégrale de la lame Oui
        Chariot porte lame Oui
        Lame interchangeable Oui
        """,
        """
        Prix unitaire TVA Désignation Qté Prix Total HTVA
        Coupeuse de plans grand format papier A0 (36 ") 04
        PRIX TOTALE HORS TVA
        """,
    ]
    metas = [
        {"source": "REPUBLIQUE_TUNISIENNE.pdf", "page": "1", "section": "deadline", "chunk_index": 0},
        {"source": "REPUBLIQUE_TUNISIENNE.pdf", "page": "3", "section": "technical", "chunk_index": 1},
        {"source": "REPUBLIQUE_TUNISIENNE.pdf", "page": "4", "section": "financial", "chunk_index": 2},
    ]
    return extract_document_facts(chunks, metas)


def test_republique_tunisienne_arabic_ocr_style_facts_are_extracted():
    facts = _republique_tunisienne_table_facts()

    assert "04 Coupeuses de plans A0" in facts["subject"]["text"]
    assert facts["deadline"]["text"] == "2025/04/10"
    assert "TUNEPS" in facts["submission_method"]["text"]

    mined = facts["mined_facts"]["text"].lower()
    assert "coupeuse de plans grand format papier a0" in mined
    assert "orientation papier a0" in mined
    assert "portrait et paysage" in mined
    assert "longueur de coupe papier a0" in mined
    assert "1190 mm" in mined


def test_extract_document_facts_handles_arabic_tender_checklist_fields():
    chunks = [
        """
        الفصل 1  :موضوع الاستشارة
        تعتزم شركة اللحوم بصفتها "المشتري العمومي" إجراء استشارة تتعلق باقتناء ووضع في طور إستخدام
        رخص مضاد للفيروسات لـ03 مستعمل و1 خادم windows server 2016.
        الفصل4 :تقديم العروض
        يتم إيداع العرض المتكون من الوثائق الإدارية والعرض الفني و المالي عبر منظومة الشراء العمومي على الخط TUNEPS.
        وقد حدد آخر أجل لقبول العروض بواسطة منظومة الشراء العمومي على الخط يوم 03 ماي 2025 على الساعة التاسعة والنصف صباحا.
        الفصل5 :صلوحية العروض
        يلتزم العارض بعرضه لمدة تسعون يوما بداية من اليوم الموالي للتاريخ الأقصى المحدد لقبول العروض.
        الفصل 6 :الوثائق المكونة لملف طلب العروض
        I. الوثائق الإداريةوتتكون من :
        1. كراس الشروط الإدارية والفنية في نسخته الأصلية مؤشر عليه الكترونيا.
        2. بطاقة إرشادات عامة حول العارض طبقا للملحق عدد 01.
        3. وثيقة الضمان المالي الوقتي بمبلغ قدره 125 دينار.
        4. نظير أصلي من السجل الوطني للمؤسسات.
        """,
        """
        II. الوثائق الخاصة بالعرض الفني:
        1. جدول الخصائص الفنية معمرة بدقة ويكون ممضى ومختوما.
        2. البطاقات الفنية Prospectus technique.
        3. إحدى شهادات المطابقة للمواصفات العالمية.
        III. وثائق الخاصة بالعرض المالي:
        1. التعهد المالي معمر بدقة ويكون ممضى ومختوما.
        2. جدول الأثمان بالدينار التونسي باحتساب جميع الأداءات.
        3. مشروع عقد الصيانة معمر بكل دقة.
        الفصل 8 :فتح العروض:
        يتم فتح العروض يوم 03 ماي 2025 على الساعة العاشرة صباحا بمقر شركة اللحوم.
        الفصل 12 :الضمان المالي النهائي:
        يجب على المزود المقبول أن يقدم ضمانا ماليا نهائيا بنسبة عشرة بالمائة (10%) من القيمة الجملية للطلبية.
        الفصل 14 :خلاص الطلبية:
        يتم تسديد مستحقات المزود في أجل أقصاه خمسة عشر يوما بعد تقديم فاتورة ومحضر التسليم الوقتي.
        الفصل15 :عقوبة التأخير
        غرامة تأخير قدرها ثلاثة بالألف (3‰) عن كل يوم تأخير.
        """,
    ]
    metas = [
        {"source": "YESNETWORKS_TECHNOLOGIES.pdf", "page": "2", "section": "general", "chunk_index": 0},
        {"source": "YESNETWORKS_TECHNOLOGIES.pdf", "page": "3", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "رخص مضاد للفيروسات" in facts["subject"]["text"]
    assert "TUNEPS" in facts["submission_method"]["text"]
    assert "03 ماي 2025" in facts["deadline"]["text"]
    assert "تسعون يوما" in facts["validity"]["text"]
    assert "فتح العروض" in facts["opening"]["text"]
    assert "125 دينار" in facts["caution"]["text"]
    assert "بطاقة إرشادات" in facts["administrative_documents"]["text"]
    assert "السجل الوطني للمؤسسات" in facts["administrative_documents"]["text"]
    assert "جدول الخصائص الفنية" in facts["technical_documents"]["text"]
    assert "جدول الأثمان" in facts["financial_documents"]["text"]
    assert "10%" in facts["definitive_caution"]["text"]
    assert "فاتورة" in facts["payment"]["text"]
    assert "غرامة تأخير" in facts["penalties"]["text"]


def test_extract_document_facts_handles_cdc_01_arabic_ocr_noise():
    chunks = [
        """
        طلب عروض عدد 2026/01 خاص باقتناء مواد اإعلامية لفائدة وزارة العدل.
        على المترشح ارسال العرض الفني والعرض المالي وكراس الشروط الإدارية والفنية والتصاريح على الشرف
        على منظومق الشراء العموميه علو الخط "توزيبس" وبفوات التاريخ والساعة المحدداز يغلق باب الإيداع.
        يتضمن الوثائق التاليقة: كراس الشروط الإدارية والفنية، العرض الفني حسب كل قسط، شهادق المطابقة
        للمواصفات الفنيق 7509001 نسخة 2015، شهادات المطابقة لمواصفات 14001 ISO، تقرير اختبار لعدد
        الصفحات، تعمير جداول الخاصيات الفنية وتقديم جذاذات فنية للمواد المطلوبة.
        العرض المالي يتضمن التعهد المالي حسب كل قسط وجدول الأشمان حسب كل قسط.
        الوثائق التي ترسل مباشرة: وشيقة الضمان الوقتي، نظير من لسجل الوطني للمؤسسات، الوشائق المثبتة
        للمؤسسات الصغرى، بطاقق الإرشادات، تصريح علو الشرف باستقلالية المؤسسة الصغرى.
        ترسل في ظرف مغلق عبر البريد مضمون الوصول أو البريد السريع أو تسلم مباشرة إلى مكتب الضبط.
        تنعقد جلسق فتح العروض وجوبا في نفس اليوم المحدد كتاريخ أقصى لقبول العروض وتجتمع لجنة فتح
        العروض في جلسق واحدق وتكون هذه الجلسة علنية.
        """,
        """
        يشترط أن يكون هذا الضمان صالح لمدق 120 يوما ابتداء من التاريخ الأقصى لقبول العروض.
        يطالب المترشح الذي تم الاحتفاظ بعرضه بتقديم ضمان نهائي يساوي 3 96 من القيمق الأصلية للصفقة
        خلال العشرين (20) يوما الموالية لإعلامه بالموافقة على الصفقة.
        مدق الضمان يجب أن لا تكون أقل مز سنق مز تاريخ القبول الوقتي ويتعهد بتعويض المواد الإعلامية
        التي بها عيوب في الصنع في أجل 7 أيام.
        يتم اعداد محضر الاستلام وامضاؤه كنتيجة لذلك. ب- الاستلام لنهائي: شريطة أن لا تكون هناك تحفظات.
        لفصل 15: غرامق الت خير تطبق عقوبة مالية على المزود على أساس واحد مز الألف (1000/01)
        عز كل يوم تأخير ولا يمكن أن تتجاوز جملة خطايا التأخير نسبة 5 96.
        الفصل 24: خلاص Jess يلتزم صاحب الصفقة بتقديم فاتورق إلى الإدارة في 4 نظائر.
        وعلى المشتري العمومي اصدار أمر بصرف المبالغ الراجعة لصاحب الصفقة في أجل ثلاثون (30) يوما
        ويتعين على المحاسب العمومي خلاص صاحب الصفقة في أجل خمس عشر (15) يوما.
        """,
    ]
    metas = [
        {"source": "CDC_01-2026.pdf", "page": "7", "section": "general", "chunk_index": 0},
        {"source": "CDC_01-2026.pdf", "page": "12", "section": "general", "chunk_index": 1},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "اقتناء مواد" in facts["subject"]["text"]
    assert "منظومة الشراء العمومية" in facts["submission_method"]["text"]
    assert "مكتب الضبط" in facts["submission_method"]["text"]
    assert "فتح العروض" in facts["opening"]["text"]
    assert "جلسة واحدة" in facts["opening"]["text"]
    assert "الضمان الوقتي" in facts["caution"]["text"]
    assert "120 يوما" in facts["caution"]["text"]
    assert "بطاقة الإرشادات" in facts["information_sheet"]["text"]
    assert "السجل الوطني للمؤسسات" in facts["rne"]["text"]
    assert "تصريح على الشرف" in facts["administrative_documents"]["text"]
    assert "ISO 9001" in facts["technical_documents"]["text"]
    assert "ISO 14001" in facts["technical_documents"]["text"]
    assert "التعهد المالي" in facts["financial_documents"]["text"]
    assert "جدول الأثمان" in facts["financial_documents"]["text"]
    assert "3 96" in facts["definitive_caution"]["text"]
    assert "20" in facts["definitive_caution"]["text"]
    assert "سنة" in facts["guarantee"]["text"]
    assert "الاستلام النهائي" in facts["reception"]["text"]
    assert "غرامة التأخير" in facts["penalties"]["text"]
    assert "1000/01" in facts["penalties"]["text"]
    assert "أمر بصرف" in facts["payment"]["text"]
    assert "30" in facts["payment"]["text"]


def test_definitive_caution_percent_ocr_does_not_match_year_ending_96():
    assert ingest._is_reliable_scalar_fact(
        "definitive_caution",
        {"text": "Le titulaire doit fournir une garantie definitive de 3 96 du montant."},
    )
    assert not ingest._is_reliable_scalar_fact(
        "definitive_caution",
        {"text": "La garantie definitive est mentionnee dans un decret de 1996 sans montant."},
    )


def test_extract_document_facts_handles_arabic_tuneps_accessories_scan_style():
    chunks = [
        """
        إستشارة 2026/31 لإقتناء لوازم أصلية لآلات سحب الأمثلة
        يعلن ديوان قيس الأراضي و المسح العقاري إجراء إستشارة لإقتناء لوازم أصلية لآلات سحب الأمثلة التالية:
        Plotwave gamme TDS CANON 3500 et (OCé 365), ROWE I4.
        لا تجوز المشاركة في هذه الاستشارة إلا عبر منظومة الشراء العمومي على الخط.
        يبقى المتعهدون ملتزمون بما قدموه من عروض لمدة (60) يوما من تاريخ آخر أجل لقبول العروض.
        يتم تقديم العروض عبر منظومة الشراء العمومي على الخط TUNEPS www.tuneps.tn.
        آخر اجل لقبول العروض 2026/05/20.
        يتم إرسال الضمان المالي الوقتي في ظرف مغلق إلى مكتب الضبط المركزي في أجل أقصاه يوم 2026/05/20.
        الفصل الثاني: الوثائق المكونة للعرض
        الشروط العامة للاستشارة تكون ممضاة ومختومة.
        شهادة تثبت أنّ المزود هو الممثل الرسمي بتونس المرخص له لبيع اللوازم.
        شهادة إثبات أصلية للوازم الأصلية Certificat d’authenticité.
        الفصل الرابع: الضمانات المالية
        الضمان المالي الوقتي بقيمة 0 دينارا صالحا لمدة 60 يوما. عدم تقديم الضمان المالي الوقتي يؤدي إلى إقصاء العرض.
        الضمان المالي النهائي مقداره ثلاثة بالمائة (903) من المبلغ الجملي للعقد.
        """,
        """
        الفصل السادس: أجل التسليم
        يسلم صاحب العقد الطلبيات في أجل أقصاه ستون (60) يوما.
        غرامات التأخير يتم احتسابها عن كل يوم تأخير وفي جميع الحالات لا يمكن أن تتجاوز جملة غرامات التأخير سقف 5% من المبلغ النهائي.
        الفصل العاشر: الإستلام الوقتي
        يتم الإستلام الوقتي بعد التثبت من مطابقة اللوازم للمواصفات وتقديم وصولات التسليم.
        الفصل الحادي عشر: الإستلام النهائي
        يتم الإستلام النهائي بعد رفع التحفظات.
        الفصل الثالث عشر: مدة الضمان
        مدة الضمان تكون لمدة (6) أشهر بداية من تاريخ الإستلام الوقتي.
        الفصل الرابع عشر: كيفية الخلاص
        يتم خلاص صاحب العقد بعد التصريح بالاستلام الوقتي وإثر تقديم فاتورة في أربعة نظائر.
        يتم إصدار الأمر بصرف المبالغ في أجل أقصاه خمسة و أربعون (45) يوما بتحويل بريدي أو بنكي.
        """,
        """
        الخصائص الفنية المطلوبة
        N° article Code Réference Proposés Tireuses de plans Plotwave et ROWE I4.
        جدول الأثمان
        السعر الفردي المبلغ الجملي دون إحتساب الآداءات مبلغ الآداءات على القيمة المضافة.
        """,
    ]
    metas = [
        {"source": "cons-31-2026-accessoires-tireuses_0001.pdf", "page": "1", "section": "general", "chunk_index": 0},
        {"source": "cons-31-2026-accessoires-tireuses_0001.pdf", "page": "3", "section": "general", "chunk_index": 1},
        {"source": "cons-31-2026-accessoires-tireuses_0001.pdf", "page": "5", "section": "technical", "chunk_index": 2},
    ]

    facts = extract_document_facts(chunks, metas)

    assert "لوازم أصلية" in facts["subject"]["text"]
    assert "سحب الأمثلة" in facts["subject"]["text"]
    assert "TUNEPS" in facts["submission_method"]["text"]
    assert "2026/05/20" in facts["deadline"]["text"]
    assert "60" in facts["validity"]["text"]
    assert "تقديم العروض" not in facts["validity"]["text"]
    assert "الضمان المالي الوقتي" in facts["caution"]["text"]
    assert "0 دينارا" not in facts["caution"]["text"]
    assert "ثلاثة بالمائة" in facts["definitive_caution"]["text"]
    assert "3%" in facts["definitive_caution"]["text"]
    assert "الشروط العامة للاستشارة" in facts["administrative_documents"]["text"]
    assert "الممثل الرسمي" in facts["administrative_documents"]["text"]
    assert "الخصائص الفنية" in facts["technical_documents"]["text"]
    assert "السعر الفردي" in facts["financial_documents"]["text"]
    assert "غرامات التأخير" not in facts["financial_documents"]["text"]
    assert "6" in facts["guarantee"]["text"]
    assert "الإستلام الوقتي" in facts["reception"]["text"]
    assert "كيفية الخلاص" not in facts["reception"]["text"]
    assert "غرامات التأخير" in facts["penalties"]["text"]
    assert "905" not in facts["penalties"]["text"]
    assert "خمسة و أربعون" in facts["payment"]["text"]


def test_mined_facts_answer_republique_tunisienne_table_questions():
    facts = _republique_tunisienne_table_facts()

    answer, _ = _answer_from_mined_facts(
        "REPUBLIQUE_TUNISIENNE.pdf",
        "Quelle est la quantité demandée ?",
        facts,
    )
    assert "4" in answer
    assert "Coupeuse de plans grand format papier A0" in answer

    answer, _ = _answer_from_mined_facts(
        "REPUBLIQUE_TUNISIENNE.pdf",
        "Quelle est la longueur de coupe minimale ?",
        facts,
    )
    assert "1190 mm" in answer

    answer, _ = _answer_from_mined_facts(
        "REPUBLIQUE_TUNISIENNE.pdf",
        "Quelle orientation papier A0 est demandée ?",
        facts,
    )
    assert "Portrait et paysage" in answer


@pytest.mark.parametrize(
    ("filename", "text", "checks"),
    [
        (
            "UBCI.pdf",
            """
            ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC, 20 Laptop et 320 ecrans.
            La premiere enveloppe interieure Enveloppe A portant la mention Offre Technique doit contenir sous peine
            de nullite les pieces suivantes : La documentation technique de chaque type de materiel propose.
            Le formulaire technique annexe au present cahier des charges. Date de commercialisation du modele propose.
            Date previsionnelle d'arret de commercialisation du modele propose. Date de fin de support du modele propose.
            La validite de la soumission. La deuxieme enveloppe interieure Enveloppe B portant la mention Offre Financiere
            doit contenir : L'offre financiere par unite et selon les quantites proposees. Prix catalogue des pieces de rechange.
            L'offre d'extension de garantie. Proposition d'un contrat cadre de service.
            ARTICLE 3. DATE LIMITE DE RECEPTION DES OFFRES : Les soumissions doivent parvenir au bureau d'ordre central.
            La date limite de reception des offres est arretee au 18 Juillet 2025, le cachet du bureau d'ordre fait foi.
            ARTICLE 13. REGLEMENT DES FACTURES : Les paiements s'effectueront par virement a la banque nommee par le
            soumissionnaire retenu. A defaut, des penalites seront fixees dans le contrat d'acquisition (5% du montant
            d'acquisition de l'article est impute par chaque jour de retard).
            """,
            {
                "subject": "160 UC",
                "deadline": "18 Juillet 2025",
                "technical_documents": "documentation technique",
                "financial_documents": "Prix catalogue",
                "payment": "virement",
                "penalties": "5%",
            },
        ),
        (
            "STEG.pdf",
            """
            Nous vous prions d'accompagner votre offre par les pieces administratives suivantes : un extrait du registre
            de commerce (original) ; un certificat d'affiliation a la CNSS (copie certifiee conforme) ; une attestation
            de situation fiscale valable a la date limite de reception des offres ; une declaration de non influence.
            Article 14 DELAI DE VALIDITE DES OFFRES Les offres techniques et financieres resteront valables durant la
            periode indiquee au DPC a compter du lendemain de la date limite de reception des offres.
            """,
            {
                "administrative_documents": "certificat d'affiliation a la CNSS",
                "cnss": "CNSS",
                "rne": "registre de commerce",
                "validity": "resteront valables",
            },
        ),
        (
            "Orange Tunisie.pdf",
            """
            Date limite de remise des offres : 30/04/2025 avant 23h45.
            Objet de la consultation : Orange Tunisie souhaite acquerir un serveur performant pour les besoins de mise
            en place d'un projet d'intelligence artificielle.
            Partie 2 : Offre financiere Votre offre financiere doit comporter : - Offre de prix detaille.
            Specificite de paiement Virement ou Traite a xxx jours apres date signature du PV de reception et de depot
            de la facture.
            """,
            {
                "subject": "serveur performant",
                "deadline": "30/04/2025",
                "financial_documents": "Offre de prix detaille",
                "payment": "Virement ou Traite",
            },
        ),
        (
            "BH ASSURANCE.pdf",
            """
            BH ASSURANCE CONSULTATION-DSI-05-2025 Objet : Consultation pour le renouvellement des Licences Veeam.
            Messieurs, La compagnie BH ASSURANCE se propose de Renouveler les Licences Veeam Backup.
            """,
            {
                "subject": "renouvellement des Licences Veeam",
            },
        ),
        (
            "BANQUE_ZITOUNA.pdf",
            """
            Cahier des Clauses administratives Particulieres
            ARTICLE 1 - OBJET DU MARCHE
            Cette consultation a pour objet l'acquisition de 150 ordinateurs de bureau avec ecran et
            20 Workstation avec ecran au titre de l'annee 2025.
            Je joins a la presente soumission le CCAP, le CCTP, une attestation fiscale et une attestation CNSS.
            """,
            {
                "subject": "150 ordinateurs de bureau",
            },
        ),
    ],
)
def test_extract_document_facts_from_representative_pdf_snippets(filename, text, checks):
    facts = _facts_from_text(filename, text)

    for field, expected_text in checks.items():
        assert field in facts
        assert expected_text.lower() in facts[field]["text"].lower()


def test_extract_document_facts_prefers_article_object_over_cover_title():
    facts = extract_document_facts(
        [
            """
            UBCI
            Cahier des Charges
            Marche d'acquisition des UC, PC Portables et des ecrans - 2025
            """,
            """
            ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC,
            20 Laptop et 320 ecrans.
            """,
        ],
        [
            {"source": "UBCI.pdf", "page": "1", "section": "general", "chunk_index": 0},
            {"source": "UBCI.pdf", "page": "2", "section": "admin", "chunk_index": 1},
        ],
    )

    assert "160 UC" in facts["subject"]["text"]
    assert "20 Laptop" in facts["subject"]["text"]
    assert facts["subject"]["page"] == "2"


def test_extract_document_facts_prefers_explicit_ubci_deadline_date():
    facts = _facts_from_text(
        "UBCI.pdf",
        """
        ARTICLE 3. DATE LIMITE DE RECEPTION DES OFFRES : Les soumissions doivent parvenir au bureau d'ordre central de
        L'UBCI a l'adresse suivante : UBCI 139 Avenue de la Liberte 1001 Tunis.
        La date limite de réception des offres est arrêtée au 18 Juillet 2025, le cachet du bureau d'ordre de L'UBCI fait foi.
        """,
        section="deadline",
    )

    assert facts["deadline"]["text"].startswith("18 Juillet 2025")


def test_extract_document_facts_resolves_steg_style_im_placeholders():
    facts = _facts_from_text(
        "STEG.pdf",
        """
        ARTICLE 1 : OBJET DU MARCHE
        Le present marche a pour objet la fourniture de (IM (1)), specifies dans le CCTP.

        ARTICLE 24 GARANTIE ET RECEPTION DEFINITIVE
        Le delai de garantie pour le materiel objet du marche est fixe a (IM (13)).

        ARTICLE 21 PENALITES DE RETARD
        Une penalite de retard de (IM (11)) par jour est appliquee. Le montant total
        ne doit pas depasser 5% du montant du marche.

        L'offre financiere doit preciser notamment : - La designation, les quantites ;
        - Les prix unitaires, les prix totaux ; - Le mode de paiement choisi ;
        - Le mode et le delai de livraison.

        ANNEXE III INSTRUCTIONS DU MARCHE
        ARTICLE 1 : OBJET DU MARCHE IM (1) Acquisition de materiel informatique
        ARTICLE 21: PENALITES DE RETARD IM (11) 0,2%
        ARTICLE 24: GARANTIE IM (13) 12 mois
        """,
    )

    assert "Acquisition de materiel informatique" in facts["subject"]["text"]
    assert "IM" not in facts["subject"]["text"]
    assert "12 mois" in facts["guarantee"]["text"]
    assert "0,2%" in facts["penalties"]["text"]
    assert "prix unitaires" in facts["financial_documents"]["text"]


def test_extract_document_facts_resolves_steg_style_dpc_placeholders():
    facts = _facts_from_text(
        "Consultation_N_2025_C020_02.pdf",
        """
        Article 1. Le present cahier a pour objet de definir les conditions de participation
        et de soumission a la consultation applicable aux travaux et/ou fourniture de biens
        et services tel que specifie dans les donnees particulieres de la consultation (DPC).

        Article 8 RECEPTION TECHNIQUE
        La reception provisoire sera prononcee en une seule fois apres la livraison des
        equipements et de la documentation technique associee. La reception definitive sera
        prononcee a l'expiration du delai de garantie.

        Article 9 GARANTIE
        A defaut d'un meilleur delai propose par le fournisseur, le delai de garantie est
        fixe a 6 mois pieces et main d'oeuvre. Ce delai commence a courir a compter de la
        date de la reception provisoire.

        Article 11 PENALITES DE RETARD
        Il sera applique une penalite de 0,2% du montant de la commande par jour calendaire
        de retard avec un maximum de 5% du montant total definitif de la commande hors TVA.

        Article 14 CAUTION BANCAIRE DE BONNE FIN
        Le titulaire doit fournir une caution bancaire a premiere demande de bonne fin.
        Le montant de cette caution doit etre egal a 5% du montant TTC de la commande.

        Article 16 CONDITIONS DE PAIEMENT
        Les factures regulierement emises sont payables a 45 jours. Mode de paiement :
        Virement Bancaire.

        Donnees particulieres de la consultation(DPC)
        Article 1 2025/C020/02 « ACQUISITION DES PIECES DE RECHANGE POUR LES IMPRIMANTES
        DE PRODUCTION CANON VP140 TUNIS ET SFAX » Objet
        (7) Elements ayant trait a l'evaluation technique et financiere devant etre
        telecharges sur TUNEPS : Le bordereau des prix dument rempli.
        (9) Extrait du registre national des entreprises.
        (10) Questionnaire technique (voir CST page 24).
        (12) Date et heure limite de reception des offres : .U../ FE ZJ/]
        (14) 90 jours.
        """,
    )

    assert "ACQUISITION DES PIECES DE RECHANGE" in facts["subject"]["text"]
    assert "CANON VP140" in facts["subject"]["text"]
    assert "TUNIS ET SFAX" in facts["subject"]["text"]
    assert "conditions de participation" not in facts["subject"]["text"].lower()
    assert "90 jours" in facts["validity"]["text"]
    assert "6 mois" in facts["guarantee"]["text"]
    assert "0,2%" in facts["penalties"]["text"]
    assert "5%" in facts["definitive_caution"]["text"]
    assert "45 jours" in facts["payment"]["text"]


def test_extract_document_facts_handles_stb_printing_solution_ocr():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        ARTICLE 1 : OBJET ET COMPOSITION DE L'APPEL D'OFFRES 1.1 Objet de l'appel d'offres
        Le present appel d'offres a pour objet l'acquisition, la fourniture, l'installation de
        equipements, materiels et logiciels, necessaires pour beneficier d'une solution d'impression
        a la banque ainsi que les prestations de Maintenance sur une Periode de trois {03} ans
        apres l'expiration de la periode de garantie. Composante 1: La fourniture, la livraison,
        l'installation, les tests de conformite et la mise en service de 50 equipements Multifonction
        impression, copie et scan. La fourniture de Consommable supplementaires necessaire pour
        imprimer >= 100 000 pages. Composante 2 : l'Application d'administration et de gestion
        d'impression pour une capacite minimale de 100 equipements.

        L'offre doit etre constituee de: des documents administratifs ci-apres : Une fiche kys
        (Know Your Supplier) etablie conformement au modele en annexe 2. Un certificat
        d'affiliation a la Caisse Nationale de Securite Sociale. L'original du certificat
        d'inscription au registre national des entreprises.

        L'offre technique ne comporte pas le formulaire de reponses dument rempli conformement
        au modele de l'annexe 3. Les justificatifs des references du soumissionnaire. Une
        documentation technique complete des equipements a fournir. Un engagement attestant la
        disponibilite des pieces de rechange. Une autorisation du constructeur en originale.
        Certification des equipements aux normes suivantes : ISO9001 et IEEE 2600.

        Le delai global des prestations d'entretien continu des equipements multifonction
        Impression-Copie-Scan est fixe a trois ans a partir de l'expiration d'une annee de garantie.
        La STB procedera au reglement du prix du marche par virement bancaire apres signature
        des PV de reception. ARTICLE 14 : PENALITES DE RETARD Les penalites de retard sont
        calculees a raison de 3%o (trois pour mille).
        """,
    )

    assert "50 équipements multifonction" in facts["subject"]["text"]
    assert "fiche kys" in facts["administrative_documents"]["text"].lower()
    assert "registre national des entreprises" in facts["administrative_documents"]["text"].lower()
    assert "documentation technique" in facts["technical_documents"]["text"].lower()
    assert "autorisation du constructeur" in facts["technical_documents"]["text"].lower()
    assert "année" in facts["guarantee"]["text"]
    assert "50 équipements multifonction" in facts["requested_items"]["text"]
    assert "100 000 pages" in facts["requested_items"]["text"]
    assert "virement bancaire" in facts["payment"]["text"]
    assert "3‰" in facts["penalties"]["text"]


def test_extract_document_facts_rejects_stb_toc_subject_fragment():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        ETCOMPOSITIONDEL'APPELD'OFFRES....., 1 = 14.. , 1 = 14

        ARTICLE 1 : OBJET ET COMPOSITION DE L'APPEL D'OFFRES
        Le present appel d'offres a pour objet l'acquisition, la fourniture,
        l'installation et la mise en service de 50 equipements multifonction
        impression-copie-scan, avec une solution d'administration et des prestations
        de maintenance.
        """,
    )

    assert "ETCOMPOSITION" not in facts["subject"]["text"]
    assert "50" in facts["subject"]["text"]
    assert "multifonction" in facts["subject"]["text"].lower()


def test_extract_document_facts_polishes_stb_subject_title_fragment():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        Etatderealisatlon/sort 'Acquisition et mise en place d'une solution
        d'impression a la STB' Page : 34/ 50
        """,
    )

    assert facts["subject"]["text"] == (
        "L'appel d'offres a pour objet l'acquisition et la mise en place "
        "d'une solution d'impression a la STB."
    )


def test_extract_document_facts_drops_caution_procedure_as_admin_documents():
    facts = _facts_from_text(
        "Societe_Tunisienne_de_Banque.pdf",
        """
        Documents administratifs :
        La caution provisoire doit etre etablie conformement au modele etabli en Annexe 1.
        La caution provisoire sera restituee aux soumissionnaires dont les offres sont eliminees.
        La caution provisoire sera mise en paiement de plein droit au profit de la STB.
        La caution provisoire sera restituee au titulaire du marche apres constitution de la caution definitive.
        Si le soumissionnaire refuse de signer le marche, la caution definitive est appelee.
        """,
    )

    assert "administrative_documents" not in facts


def test_extract_document_facts_builds_tender_profile():
    facts = _facts_from_text(
        "UBCI.pdf",
        """
        ARTICLE 1 : OBJET DE LA CONSULTATION : L'UBCI se propose d'acquerir 160 UC,
        20 Laptop et 320 ecrans.
        La date limite de reception des offres est arretee au 18 Juillet 2025.
        Les soumissions doivent parvenir au bureau d'ordre central de l'UBCI.
        L'offre financiere par unite et selon les quantites proposees.
        """,
    )

    profile = facts["tender_profile"]

    assert profile["schema"] == "tender_profile.v1"
    assert "object" in profile["fields"]
    assert "deadline" in profile["fields"]
    assert profile["fields"]["object"]["page"] == facts["subject"]["page"]
    assert "160 UC" in profile["fields"]["object"]["text"]
    assert profile["coverage"]["core_present"] >= 3
    assert "payment" in profile["coverage"]["missing_core_fields"]


def test_extract_document_facts_supports_tender_checklist_fields():
    facts = _facts_from_text(
        "CDC.pdf",
        """
        Les soumissions doivent parvenir par voie postale ou par depot direct au bureau d'ordre central.
        Les variantes ne sont pas autorisees.
        Le dossier administratif comprend une fiche de renseignements, une attestation de situation fiscale,
        une attestation d'affiliation a la CNSS et un extrait du registre de commerce.
        L'offre technique doit contenir une autorisation du constructeur et une liste des references similaires.
        La reception provisoire sera prononcee apres installation et la reception definitive apres garantie.
        Une caution definitive de 10% du montant du marche est exigee.
        """,
    )

    assert "voie postale" in facts["submission_method"]["text"]
    assert "ne sont pas autorisees" in facts["variants"]["text"]
    assert "fiche de renseignements" in facts["information_sheet"]["text"]
    assert "situation fiscale" in facts["fiscal_certificate"]["text"]
    assert "CNSS" in facts["cnss"]["text"]
    assert "registre de commerce" in facts["rne"]["text"]
    assert "autorisation du constructeur" in facts["manufacturer_authorization"]["text"]
    assert "references similaires" in facts["references"]["text"]
    assert "reception provisoire" in facts["reception"]["text"]
    assert "caution definitive" in facts["definitive_caution"]["text"]


def test_extract_document_facts_handles_tunisian_saudi_bank_article_layout():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        ARTICLE 0. OBJET DU DOSSIER Ils fixent les procédures de l'appel d'offres et stipulent
        les conditions du marché.

        ARTICLE 1. OBJET DU MARCHE La TSB envisage de mettre a niveau l'Infrastructure Systeme
        (Site Principal et de Backup a Kairouan). A cet effet la TSB lance le present appel d'offres
        en lot unique pour l'acquisition, la mise en place et la migration des serveurs et de la solution
        de virtualisation VMWARE.

        ARTICLE 7. VALIDITE DES OFFRES Les offres demeureront valables pour une periode de 90 jours
        apres la date limite de reception des offres fixee par TSB.

        ARTICLE 14. CAUTIONNEMENT 1. Caution provisoire Chaque offre doit etre accompagnee d'une
        caution bancaire provisoire. Le montant de la caution provisoire s'elevera a 12.000 DT.
        2. Caution definitive Le titulaire du marche devra fournir une caution definitive d'une valeur
        egale a 3% du montant total du marche toutes taxes comprises.

        ARTICLE 16. PRESENTATION & RECEPTION DE LA SOUMISSION
        1. DOSSIER ADMINISTRATIF : La caution bancaire provisoire. RNE recent valable a la date
        d'ouverture des offres. Une attestation d'affiliation a la CNSS.
        2. OFFRE TECHNIQUE Le dossier de l'offre technique doit contenir les pieces suivantes :
        La liste des equipements. Documentation techniques. Engagement concernant l'origine des fournitures.
        La certification sur HPE Synergy. La certification sur Vmware.
        3. OFFRE FINANCIERE Le dossier de l'offre financiere doit obligatoirement comporter :
        La lettre de soumission. Le bordereau des prix. Le recapitulatif des prix.

        ARTICLE 17. DATE LIMITE DE RECEPTION DES OFFRES Les offres doivent parvenir par voie postale
        ou remise directement au bureau d'ordre de T.S.B. La date limite de la réception des offres est
        fixée au 24 février 2025.

        ARTICLE 26. GARANTIE Le délai de garantie est de 3 (trois) ans a compter de la date de la reception
        provisoire sans reserve.
        ARTICLE 28. RECEPTION La reception provisoire sera prononcee apres essais satisfaisants.
        La reception definitive sera prononcee un an apres la reception provisoire sans reserves.
        ARTICLE 29. CONDITIONS DE PAIEMENT Les conditions de paiement sont fixees comme suit :
        20% a la livraison du materiel. 50% a la finalisation du site principal. 20% a la finalisation
        du site de backup. 10% retenue de garantie.
        ARTICLE 30. MODALITES DE PAIEMENT La facture sera payable par chèque ou virement bancaire
        dans un delai de 30 jours.
        ARTICLE 34. PENALITE DE RETARD Le fournisseur devra payer une penalite calculee a raison
        d'un pour mille pour chaque jour de retard. Le montant total de la penalite ne doit pas exceder
        cinq pour cent de la valeur totale du marche.
        """,
    )

    assert "Infrastructure Systeme" in facts["subject"]["text"]
    assert facts["deadline"]["text"] == "24 février 2025"
    assert "90 jours" in facts["validity"]["text"]
    assert "12.000 DT" in facts["caution"]["text"]
    assert "voie postale" in facts["submission_method"]["text"]
    assert "RNE" in facts["administrative_documents"]["text"]
    assert "CNSS" in facts["administrative_documents"]["text"]
    assert "Documentation techniques" in facts["technical_documents"]["text"]
    assert "Vmware" in facts["technical_documents"]["text"]
    assert "lettre de soumission" in facts["financial_documents"]["text"].lower()
    assert "3 (trois) ans" in facts["guarantee"]["text"]
    assert "reception definitive" in facts["reception"]["text"].lower()
    assert "50%" in facts["payment"]["text"]
    assert "pour mille" in facts["penalties"]["text"]


def test_extract_document_facts_rejects_correspondence_as_submission_method():
    facts = _facts_from_text(
        "CORRESPONDENCE.pdf",
        """
        Les demandes d'eclaircissement doivent parvenir par courrier electronique au secretariat.
        Toute correspondance doit etre envoyee par voie postale.

        Les offres doivent parvenir sous pli ferme au bureau d'ordre central avant la date limite.
        """,
    )

    assert "pli ferme" in facts["submission_method"]["text"]
    assert "eclaircissement" not in facts["submission_method"]["text"].lower()


def test_extract_document_facts_rejects_execution_validity_for_offer_validity():
    facts = _facts_from_text(
        "VALIDITY.pdf",
        """
        ARTICLE 4 VALIDITE DU CONTRAT
        La validite du contrat couvre toute la periode d'execution du marche.

        ARTICLE 5 VALIDITE DES OFFRES
        Les offres resteront valables pendant 120 jours a compter de la date limite de reception.
        """,
    )

    assert "120 jours" in facts["validity"]["text"]
    assert "execution" not in facts["validity"]["text"].lower()


def test_extract_document_facts_penalizes_subject_toc_and_forms():
    facts = _facts_from_text(
        "SUBJECT.pdf",
        """
        ARTICLE 1 OBJET ........................................ 3
        ARTICLE 2 CONDITIONS ................................... 4
        ARTICLE 3 VALIDITE ..................................... 5

        ANNEXE N 1 MODELE DE SOUMISSION Objet : formulaire de reponse.

        Article 1 Objet du marche
        Le present appel d'offres a pour objet l'acquisition et la mise en place
        d'une solution de sauvegarde centralisee.
        """,
    )

    assert "solution de sauvegarde centralisee" in facts["subject"]["text"]
    assert "ARTICLE 1 OBJET" not in facts["subject"]["text"]


def test_extract_document_facts_handles_bct_article_tender_fields():
    facts = _facts_from_text(
        "soumissionner_ARTICLE_2026-03-05.pdf",
        """
        SOMMAIRE
        ARTICLE1, 1 = Objet du marche
        ARTICLE2, 1 = : Composition du marche
        ARTICLE3, 1 = Pieces constitutives du dossier

        ARTICLE 1ER : OBJET DU MARCHE
        Le present marche a pour objet de definir les conditions generales et speciales
        pour la fourniture et la livraison d'imprimes simples, de fournitures de bureaux,
        de fournitures informatiques, de fournitures d'imprimerie et de fournitures de caisse.

        ARTICLE 4 : PRESENTATION DES OFFRES
        Une sous-enveloppe fermee pour l'offre financiere contenant :
        La soumission. Le bordereau des prix. Le sous-detail des prix par lot.
        Le cautionnement provisoire de 1,5% du montant de la soumission doit etre joint.
        Les offres doivent etre adressees par voie postale ou deposees au Bureau d'Ordre Central
        au plus tard le 07/07/2025 a 12h00.

        ARTICLE 5 : DELAI DE VALIDITE DES OFFRES
        Les soumissionnaires sont engages par leurs offres pendant 120 jours a compter
        de la date limite fixee pour la reception des plis.

        Toute enveloppe comportant une reference relative au nom du soumissionnaire est automatiquement rejetee.
        """,
    )

    assert facts["subject"]["text"].startswith("Le present marche a pour objet") or facts["subject"][
        "text"
    ].startswith("Le présent marché a pour objet")
    assert "fourniture et la livraison d'imprimes simples" in facts["subject"]["text"]
    assert "Composition du marche" not in facts["subject"]["text"]
    assert "120 jours" in facts["validity"]["text"]
    assert "reception des plis" in facts["validity"]["text"].lower()
    assert "1,5%" in facts["caution"]["text"]
    assert "Bureau d'Ordre Central" in facts["submission_method"]["text"]
    assert "07/07" not in facts["submission_method"]["text"]
    assert "La soumission" in facts["financial_documents"]["text"]
    assert "bordereau des prix" in facts["financial_documents"]["text"].lower()
    assert "sous-detail des prix" in facts["financial_documents"]["text"].lower()
    assert "references" not in facts


def test_extract_document_facts_keeps_designation_column_from_ocr_tables():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        1. DOSSIER ADMINISTRATIF : N° de la pièce, 1 = Désignations. N° de la pièce, 2 = Authentifications.
        1, 1 = La caution bancaire provisoire d'un montant égal à Douze mille (12 000) Dinars.
        1, 2 = Cachet signature du soumissionnaire.
        2, 1 = joint en annexe(2). Les tableaux portant sur les références, clairement remplis.
        2, 2 = Dûment signé paraphé et daté par soumissionnaire.
        3, 1 = Déclaration sur l'honneur concernant l'exactitude des informations fournies selon le modèle joint en annexe(7).
        3, 2 = Dûment signé paraphé et daté par soumissionnaire.

        2. OFFRE TECHNIQUE Le dossier de l'offre technique doit contenir les pieces suivantes :
        Documentation technique. La certification sur Vmware.
        """,
    )

    admin_text = facts["administrative_documents"]["text"]
    assert "La caution bancaire provisoire" in admin_text
    assert "Les tableaux portant sur les références" in admin_text
    assert "Déclaration sur l'honneur" in admin_text
    assert "Authentifications" not in admin_text
    assert "Cachet signature" not in admin_text
    assert "Dûment signé" not in admin_text


def test_extract_document_facts_keeps_financial_designations_from_ocr_tables():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        3. OFFRE FINANCIERE Le dossier de l'offre financière doit obligatoirement comporter :
        1, Désignations = La lettre de soumission conformément au modèle joint en annexe(1).
        1, Authentifications = Date, signature et cachet du soumissionnaire.
        2, Désignations = Le bordereau des prix conformément au modèle joint en annexe(10).
        2, Authentifications = Date, signature et cachet du soumissionnaire.
        3, Désignations = Le récapitulatif des prix conformément au modèle joint en annexe(11).
        3, Authentifications = Date, signature et cachet du soumissionnaire.
        """,
    )

    financial_text = facts["financial_documents"]["text"]
    assert "La lettre de soumission" in financial_text
    assert "Le bordereau des prix" in financial_text
    assert "Le récapitulatif des prix" in financial_text
    assert "Authentifications" not in financial_text
    assert "signature et cachet" not in financial_text


def test_extract_document_facts_strips_ocr_prefixes_from_designation_items():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        3. OFFRE FINANCIERE Le dossier de l'offre financière doit obligatoirement comporter :
        1, Désignations = a , oa La lettre de soumission conformément au modèlejointenannexe(1).
        1, Authentifications = Date, signature et cachet du soumissionnaire.
        2, Désignations = , un Le bordereau des prix Conformément au modèlejointenannexe(10).
        2, Authentifications = Date, signature et cachet du soumissionnaire.
        3, Désignations = Ve . . , oo. Le récapitulatif des prix. Conformément au modèle joint enannexe(11).
        3, Authentifications = Date, signature et cachet du soumissionnaire.

        1. DOSSIER ADMINISTRATIF :
        5, 1 = ' ' 4; PrésentationduSoumissionnaire.
        5, 2 = Dûment signé paraphé et daté par soumissionnaire.
        """,
    )

    financial_text = facts["financial_documents"]["text"]
    assert "- La lettre de soumission" in financial_text
    assert "- Le bordereau des prix" in financial_text
    assert "- Le récapitulatif des prix" in financial_text
    assert "modèle joint en annexe(1)" in financial_text
    assert "annexe(11)" in financial_text
    assert "a , oa" not in financial_text
    assert ", un Le" not in financial_text
    assert "Ve . ." not in financial_text

    admin_text = facts["administrative_documents"]["text"]
    assert "Présentation du Soumissionnaire" in admin_text
    assert "' ' 4" not in admin_text


def test_extract_document_facts_handles_messy_tsb_ocr_fragments():
    facts = _facts_from_text(
        "TUNISIAN_SAUDI_BANK.pdf",
        """
        Les offres doivent obligatoirement parvenir par voie postale recommandee ou par rapide-poste
        ou remise directement au bureau d'ordre de T.S.B (cachet du bureau d'ordre faisant foi) a
        l'adresse suivante. La date limite de la réception des offres est fixée au 24 février 2025.

        La caution bancaire provisoire d'un montant égal 4 Douze mille (12 000) Dinars en original
        et établi conformément aux dispositions du cahier des charges.

        L' offre technique et l'offre financière doivent être placées dans deux enveloppes séparées.
        Le dossier de l' offre technique doit contenir sous peine de nullité les pièces suivantes :
        La liste de l'équipe intervenante accompagnée de leur CV et copie des diplômes et des certifications.
        Documentation technique. Engagement concernant l'origine des fournitures.
        La certification sur HPE Synergy. La certification sur Vmware.
        Le dossier de l'offre financière doit obligatoirement comporter : La lettre de soumission.
        """,
    )

    assert "voie postale" in facts["submission_method"]["text"]
    assert "bureau d'ordre" in facts["submission_method"]["text"]
    assert "Douze mille" in facts["caution"]["text"]
    assert "12 000" in facts["caution"]["text"]
    assert "Documentation technique" in facts["technical_documents"]["text"]
    assert "Vmware" in facts["technical_documents"]["text"]


def test_extract_document_facts_handles_cetime_dense_consultation_articles():
    facts = _facts_from_text(
        "CETIME.pdf",
        """
        Article 1. Objet de la consultation : Le present cahier des charges a pour objet
        l'accompagnement a la mise en place d'un systeme de gestion des documents et des archives,
        Article 2. CONDITION DE SOUMISSION : La participation a la consultation est ouverte
        a toutes les personnes physiques ou morales etablies en Tunisie.

        Pieces administratives a fournir :
        + Une fiche de renseignement.
        + Le present cahier des charges paraphe, signe et cachete avec la mention lu et approuve.
        + un original du registre national de l'entreprise.
        + Attestation de la situation fiscale valide.
        + CV des intervenants.
        + Une declaration de non faillite.

        Les candidats sont lies par leurs offres pour une periode de soixante jours (60) jours
        a compter du jour suivant la date limite fixee pour la reception des offres.
        Toute offre ne contenant pas la liste d'au moins 3 travaux similaires durant les cinq
        dernieres annees sera rejetee.

        La reception est prononcee suite a: La validation par le CETIME des prestations requises;
        La fin de la formation; La remise de la documentation technique. Le PV de reception doit
        etre signe par les deux parties sans reserves.
        Le reglement est effectue par virement suite au depot de la facture au bureau d'ordre
        central du CETIME et a la fourniture du PV signe par les deux parties sans reserves.
        """,
    )

    assert "systeme de gestion des documents et des archives" in facts["subject"]["text"]
    assert "participation" not in facts["subject"]["text"].lower()
    assert "soixante jours" in facts["validity"]["text"]
    assert "fiche de renseignement" in facts["administrative_documents"]["text"].lower()
    assert "registre national" in facts["administrative_documents"]["text"].lower()
    assert "situation fiscale" in facts["administrative_documents"]["text"].lower()
    assert "non faillite" in facts["administrative_documents"]["text"].lower()
    assert "3 travaux similaires" in facts["references"]["text"]
    assert "PV de reception" in facts["reception"]["text"]
    assert "virement" in facts["payment"]["text"].lower()


def test_extract_document_facts_handles_tunisie_telecom_consumables_consultation():
    facts = _facts_from_text(
        "Tunisie_Telecom_DCSI.pdf",
        """
        ARTICLE 1-2 : OBJET
        La presente consultation a pour objet la conclusion d'un marche cadre pour l'acquisition des
        consommables et accessoires informatiques, au profit Tunisie Telecom, dont les specifications
        techniques sont definies ci-apres dans le cahier des charges des clauses techniques.
        Cette acquisition est repartie en trois (03) lots separes comme suit:
        N Designation Ref Qte MIN Qte MAX Lot N01 Consommables imprimante HP Toner d'origine HP.

        ARTICLE 3.2 : DOCUMENTS CONSTITUTIFS DE L'OFFRE
        Les offres contiennent les pieces administratives suivantes :
        Presentation du Soumissionnaire.
        Une attestation d'affiliation a la CNSS.
        Extrait du registre national des entreprises actualisees datant de moins de 30 jours.
        Le cahier des charges signe paraphe par le soumissionnaire.
        Attestation constructeur attestant l'originalite des produits.

        A- DOSSIER DE L'OFFRE FINANCIERE
        Le(s) lettre(s) de soumission pour chaque lot(s).
        Devis estimatif detaille : le soumissionnaire est tenu d'indiquer les quantites
        et les prix unitaires (PU) de chaque article et le prix total (PT).

        B- DOSSIER DE L'OFFRE TECHNIQUE
        La deuxieme enveloppe doit porter la mention Offre technique Consultation N 02/DCSI/2025
        et sera composee des pieces techniques suivantes dans l'ordre indique :
        Presentation de l'offre technique.
        Presentation des specifications techniques conformement aux tableaux des clauses techniques particuliers.
        Les delais de livraison des articles.

        ARTICLE 4.1 : DATE LIMITE DE RECEPTION DES OFFRES
        Les soumissionnaires doivent disposer leurs offres au bureau d'ordre a l'adresse suivante :
        TUNISIE TELECOM Direction Centrale des Systemes d'Information Les Jardins du Lac.
        Au plus tard, le 10/06/2025. La date et le numero d'enregistrement sur le registre
        du bureau d'ordre de TUNISIE TELECOM faisant foi.

        ARTICLE 4.2 : DELAI DE VALIDITE DE L'OFFRE
        Les offres seront valables pendant 90 jours a compter de la date limite de reception des offres.

        ARTICLE 5.1 : OUVERTURE DES PLIS
        L'ouverture des plis aura lieu dans les locaux de TUNISIE TELECOM.

        ARTICLE 9 : RECEPTION PROVISOIRE-RECEPTION DEFINITIVE
        9.1 Reception quantitative. 9.2 Reception provisoire. 9.3 Reception definitive.

        ARTICLE 13 : GARANTIE
        Tunisie Telecom informera le fournisseur pour le remplacement dans un delai de 48 heures
        des articles defectueux pendant une periode de 6 mois a partir de la date de la reception
        provisoire pour les Lots 1 et 2. Pour le Lot 3 : la garantie est de 2 ans pour les
        casques et souris sans fil et 3 annees pour les douchettes.

        ARTICLE 14 : CONDITIONS DE PAIEMENT
        Pour chaque Appel de commande, Le paiement se fera 100% sera regle a 60 jours sur
        presentation de l'originale de la facture et de(s) bon(s) de livraison et le PV de
        reception provisoire.

        ARTICLE 15 : PENALITES POUR RETARD
        Il sera applique une penalite pour retard de cinq pour mille (5‰) par jour sur le
        montant des articles non livres avec un maximum de 10% du montant definitif du marche.
        """,
    )

    assert "marche cadre" in facts["subject"]["text"].lower()
    assert "consommables et accessoires informatiques" in facts["subject"]["text"].lower()
    assert "toner" not in facts["subject"]["text"].lower()
    assert "bureau d'ordre" in facts["submission_method"]["text"].lower()
    assert "tunisie telecom" in facts["submission_method"]["text"].lower()
    assert "10/06/2025" in facts["deadline"]["text"]
    assert "90 jours" in facts["validity"]["text"]
    assert "locaux de TUNISIE TELECOM" in facts["opening"]["text"]
    assert "soumissionnaire" in facts["administrative_documents"]["text"].lower()
    assert "cnss" in facts["administrative_documents"]["text"].lower()
    assert "registre national" in facts["administrative_documents"]["text"].lower()
    assert "originalite des produits" in facts["manufacturer_authorization"]["text"].lower()
    assert "lettre" in facts["financial_documents"]["text"].lower()
    assert "devis estimatif" in facts["financial_documents"]["text"].lower()
    assert "presentation de l'offre technique" in facts["technical_documents"]["text"].lower()
    assert "specifications techniques" in facts["technical_documents"]["text"].lower()
    assert "6 mois" in facts["guarantee"]["text"]
    assert "2 ans" in facts["guarantee"]["text"]
    assert "3 annees" in facts["guarantee"]["text"]
    assert "Reception quantitative" in facts["reception"]["text"]
    assert "60 jours" in facts["payment"]["text"]
    assert "facture" in facts["payment"]["text"].lower()
    assert "cinq pour mille" in facts["penalties"]["text"].lower()
    assert "10%" in facts["penalties"]["text"]


def test_extract_document_facts_opening_handles_huis_clos_and_commission_sections():
    bfpm_e = _facts_from_text(
        "bfpm-e.pdf",
        """
        Atticle 8. OUVERTURE DES PLIS
        a) La commission des consultations se reunit en seance unique (a huis clos)
        pour ouvrir les enveloppes contenant les offres techniques et financieres.
        b) La date de l'ouverture des plis techniques et financiers doit avoir lieu
        dans un delai maximum d'un jour ouvrable suivant la date limite de reception des offres.
        """,
    )

    assert "huis clos" in bfpm_e["opening"]["text"].lower()
    assert "offres techniques" in bfpm_e["opening"]["text"].lower()

    cc_cnss = _facts_from_text(
        "cc_cnss.docx",
        """
        Ouverture des plis

        Pendant cette seance la commission d'ouverture des offres procedera a l'ouverture
        simultanement des enveloppes parvenues dans les delais au bureau d'ordre central
        et le decryptage des offres parvenues en ligne.

        Montant des offres
        Les montants doivent etre presentes hors taxes et TTC.
        """,
    )

    assert "commission d'ouverture" in cc_cnss["opening"]["text"].lower()
    assert "decryptage des offres" in cc_cnss["opening"]["text"].lower()
    assert "montants doivent etre presentes" not in cc_cnss["opening"]["text"].lower()


def test_extract_document_facts_references_handles_hpe_frame_reference_table():
    facts = _facts_from_text(
        "cdc_hpe.pdf",
        """
        ARTICLE 4 - Soumissionnaire et Equipe intervenante :
        Soumissionnaire : N° Designation Exigence minimale (*)
        1 Nombre de reference 3 References dans l'installation des frames HPE
        durant les 3 dernieres annees (avec justificatifs)
        2 Nombre d'effectif dedie aux projets d'installation, de configuration
        et de maintenance des Frames HPE 2
        (*) Seules les references justifiees par une commande, un contrat ou facture
        seront prises en consideration.
        """,
    )

    assert "frames HPE" in facts["references"]["text"]
    assert "justificatifs" in facts["references"]["text"].lower()


def test_extract_document_facts_submission_method_handles_address_delivery_clause():
    facts = _facts_from_text(
        "cdc_hpe.pdf",
        """
        ARTICLE 8 - DATE LIMITE DE RECEPTION DES OFFRES
        Les soumissionnaires doivent envoyer leurs offres a l'adresse suivante :
        BANQUE ZITOUNA 02 Boulevard Qualite de la vie - LE KRAM - TUNIS.
        La date limite de reception des offres est arretee au 06/04/2026.
        Le registre du bureau d'ordre de la BANQUE ZITOUNA faisant foi.
        """,
    )

    assert "adresse suivante" in facts["submission_method"]["text"].lower()
    assert "BANQUE ZITOUNA" in facts["submission_method"]["text"]
    assert "LE KRAM" in facts["submission_method"]["text"]


def test_extract_document_facts_reception_and_payment_handle_cnss_sections():
    facts = _facts_from_text(
        "cc_cnss.docx",
        """
        Reception provisoire
        Apres la mise en exploitation des composantes de l'offre, il sera procede
        au cours d'une periode de deux mois a l'utilisation en reel des solutions installees.
        A l'issue de cette periode et si aucune anomalie n'est constatee, un proces-verbal
        de reception provisoire sera etabli par les 2 parties.

        Reception definitive
        La reception definitive sera prononcee apres quatre mois de la reception provisoire
        et donnera lieu a un proces-verbal de reception definitive.

        Modalites de paiement
        Le reglement se fera integralement apres la signature du proces-verbal de la reception
        definitive sans reserve. La facture doit etre deposee au Bureau d'Ordre Central de
        la CNSS. La facture emises par le fournisseur est payable a 45 jours de la date de
        reception de la facture par la CNSS.
        """,
    )

    assert "reception provisoire" in facts["reception"]["text"].lower()
    assert "reception definitive" in facts["reception"]["text"].lower()
    assert "45 jours" in facts["payment"]["text"]
    assert "Bureau d'Ordre Central" in facts["payment"]["text"]


def test_extract_document_facts_classifies_contenu_offre_table():
    facts = _facts_from_text(
        "ATI.pdf",
        """
        ARTICLE 1 : OBJET DE LA PRESENTE CONSULTATION
        L'Agence Tunisienne d'Internet se propose de lancer une consultation pour
        l'acquisition de materiels informatiques repartis en 05 lots independants.

        ARTICLE 2 : PRESENTATION DES OFFRES
        L'offre est placee dans une enveloppe contenant outre les documents administratifs
        et techniques, la soumission ainsi que le bordereau detaille des prix. Les offres
        devront parvenir par voie postale ou par remise directe au bureau d'ordre.

        ARTICLE 3 : CONTENU DE L'OFFRE
        L'offre doit comporter les pieces suivantes :
        N DESIGNATION AUTHENTIFICATION PIECE
        1 Fiche de renseignements generaux sur le soumissionnaire dument complete Selon le modele figurant en annexe n1
        2 Cahier des charges paraphe sur chaque page date signature et cachet du soumissionnaire
        3 Tableaux des specificites techniques dument remplis et signes et les documents techniques y afferant
        4 La soumission dument remplie et signee par le soumissionnaire Selon le modele figurant en annexe n2
        5 Les Bordereaux des prix dument remplis et signes par le soumissionnaire Selon le modele figurant en annexe n3
        6 Engagement de garantie complete
        7 Un extrait du registre de commerce/certificat RNE

        ARTICLE 4 : SPECIFICATION TECHNIQUE
        Garantie 1an.
        """,
    )

    assert "fiche de renseignements" in facts["administrative_documents"]["text"].lower()
    assert "cahier des charges" in facts["administrative_documents"]["text"].lower()
    assert "registre de commerce" in facts["administrative_documents"]["text"].lower()
    assert "tableaux des specificites techniques" in facts["technical_documents"]["text"].lower()
    assert "documents techniques y afferant" in facts["technical_documents"]["text"].lower()
    assert "soumission" in facts["financial_documents"]["text"].lower()
    assert "bordereaux des prix" in facts["financial_documents"]["text"].lower()


def test_tender_checklist_answer_uses_extracted_facts():
    answer = build_tender_checklist_answer(
        "UBCI.pdf",
        {
            "subject": {"text": "L'UBCI se propose d'acquerir 160 UC", "page": "2", "section": "admin"},
            "deadline": {"text": "18 Juillet 2025", "page": "3", "section": "deadline"},
            "variants": {"text": "Les variantes ne sont pas autorisees", "page": "4", "section": "admin"},
        },
    )

    assert "Analyse de consultation" in answer
    assert "L'UBCI se propose d'acquerir 160 UC" in answer
    assert "18 Juillet 2025" in answer
    assert "Non - Les variantes ne sont pas autorisees" in answer
    assert "Non mentionne dans ce document." in answer


def test_language_detection_defaults_to_french_for_tender_questions():
    assert _detect_answer_language("Une attestation d'affiliation a la CNSS est-elle exigee ?") == "fr"
    assert _detect_answer_language("Un extrait du registre de commerce est-il exige ?") == "fr"
    assert _detect_answer_language("Existe-t-il des penalites de retard ?") == "fr"
    assert _language_instruction("RNE ?") == "Reponds uniquement en francais."


def test_meta_language_answers_are_hallucination_signals():
    answer = "The user requested the response to be in English."

    assert any(signal.lower() in answer.lower() for signal in HALLUCINATION_SIGNALS)


@pytest.mark.asyncio
async def test_facts_first_answers_new_scalar_fields(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(
        filename="CDC 01-2026.pdf",
        status="indexed",
        extracted_facts={
            "caution": {
                "text": "5 000 DT",
                "page": "6",
                "section": "guarantee",
            },
            "validity": {
                "text": "120 jours a compter de la date limite de reception des offres",
                "page": "8",
                "section": "deadline",
            },
        },
    )

    async with initialized_db.session_factory() as session:
        answer, metas = await answer_from_document_facts(
            db=session,
            question="Quelle est la caution provisoire ?",
            source_filter=["CDC 01-2026.pdf"],
            department_filter=["commerciale"],
            universe_id=None,
            user_id="user-1",
            is_admin=True,
        )

    assert "Caution : 5 000 DT" in answer
    assert "Source: CDC 01-2026.pdf, page 6." in answer
    assert metas == [
        {
            "source": "CDC 01-2026.pdf",
            "page": "6",
            "section": "guarantee",
            "score": 1.0,
        }
    ]


@pytest.mark.asyncio
async def test_facts_first_answers_lobjet_question_with_apostrophe(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(
        filename="UBCI.pdf",
        status="indexed",
        extracted_facts={
            "subject": {
                "text": "L'UBCI se propose d'acquerir 160 UC, 20 Laptop et 320 ecrans",
                "page": "2",
                "section": "admin",
            }
        },
    )

    async with initialized_db.session_factory() as session:
        answer, metas = await answer_from_document_facts(
            db=session,
            question="Quel est l'objet de la consultation ?",
            source_filter=["UBCI.pdf"],
            department_filter=["commerciale"],
            universe_id=None,
            user_id="user-1",
            is_admin=True,
        )

    assert "160 UC" in answer
    assert "Source: UBCI.pdf, page 2." in answer
    assert metas[0]["source"] == "UBCI.pdf"


@pytest.mark.asyncio
async def test_facts_first_answers_structured_document_lists(
    initialized_db,
    seed_department,
    seed_user,
    seed_document,
):
    await seed_department()
    await seed_user()
    await seed_document(
        filename="STEG.pdf",
        status="indexed",
        extracted_facts={
            "administrative_documents": {
                "text": "- un extrait du registre de commerce\n- un certificat d'affiliation a la CNSS",
                "items": [
                    {
                        "text": "un extrait du registre de commerce",
                        "page": "1",
                        "section": "admin",
                    },
                    {
                        "text": "un certificat d'affiliation a la CNSS",
                        "page": "1",
                        "section": "admin",
                    },
                ],
                "page": "1",
                "section": "admin",
            }
        },
    )

    async with initialized_db.session_factory() as session:
        answer, metas = await answer_from_document_facts(
            db=session,
            question="Quels documents administratifs faut-il fournir ?",
            source_filter=["STEG.pdf"],
            department_filter=["commerciale"],
            universe_id=None,
            user_id="user-1",
            is_admin=True,
        )

    assert "Documents administratifs :" in answer
    assert "- un extrait du registre de commerce" in answer
    assert "- un certificat d'affiliation a la CNSS" in answer
    assert "Source: STEG.pdf, page 1." in answer
    assert metas == [
        {
            "source": "STEG.pdf",
            "page": "1",
            "section": "admin",
            "score": 1.0,
        }
    ]
