#!/usr/bin/env python3
"""
Update thesis .docx: text fixes + figure insertion.
Run from project root: python experiments/update_thesis.py
"""

import copy
import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).parent.parent
DOCX = ROOT / "docs/thesis/Tesis_Churata_Apaza_2025_APA6_FINAL.docx"
BACKUP = ROOT / "docs/thesis/Tesis_Churata_Apaza_2025_APA6_FINAL.backup2.docx"
FIGS = ROOT / "outputs/figures"


# ── Low-level helpers ──────────────────────────────────────────────────────────

def delete_para(para):
    p = para._element
    p.getparent().remove(p)


def replace_para_text(para, new_text):
    """Replace paragraph content, preserving first run formatting."""
    bold = para.runs[0].bold if para.runs else None
    italic = para.runs[0].italic if para.runs else None
    font_size = para.runs[0].font.size if para.runs else None
    for run in para.runs:
        run.text = ""
    r = para.runs[0] if para.runs else para.add_run()
    r.text = new_text
    if bold is not None:
        r.bold = bold
    if italic is not None:
        r.italic = italic
    if font_size is not None:
        r.font.size = font_size


def find_para(doc, text_fragment):
    """Return first paragraph containing text_fragment, or None."""
    for para in doc.paragraphs:
        if text_fragment in para.text:
            return para
    return None


def insert_para_after(doc, anchor_para, text):
    """Insert a plain paragraph immediately after anchor_para."""
    tmp = doc.add_paragraph(text)
    xml = copy.deepcopy(tmp._element)
    tmp._element.getparent().remove(tmp._element)
    anchor_para._element.addnext(xml)


def insert_figure_after(doc, anchor_text, image_path, caption_text, width_inches=5.5):
    """
    Insert image paragraph + APA caption paragraph immediately after
    the first paragraph containing anchor_text.
    Order: anchor → image → caption.
    """
    anchor = find_para(doc, anchor_text)
    if anchor is None:
        print(f"  [WARN] Anchor not found: {anchor_text[:70]!r}")
        return False

    # Build image + caption at end of doc (easiest way to get valid XML)
    tmp_img = doc.add_paragraph()
    tmp_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tmp_img.add_run().add_picture(str(image_path), width=Inches(width_inches))

    tmp_cap = doc.add_paragraph(caption_text)
    tmp_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in tmp_cap.runs:
        r.italic = True
        r.font.size = Pt(10)

    img_xml = copy.deepcopy(tmp_img._element)
    cap_xml = copy.deepcopy(tmp_cap._element)
    tmp_img._element.getparent().remove(tmp_img._element)
    tmp_cap._element.getparent().remove(tmp_cap._element)

    # addnext inserts immediately after; insert cap first so img ends up between anchor and cap
    anchor._element.addnext(cap_xml)
    anchor._element.addnext(img_xml)

    print(f"  [OK] {Path(image_path).name} → after {anchor_text[:55]!r}")
    return True


# ── Step 3: Text changes ───────────────────────────────────────────────────────

def step3a_delete_stale_para(doc):
    """Delete the standalone paragraph with stale data (63.58%, 85.66%)."""
    print("\n[3a] Deleting stale data paragraph (63.58%)...")
    # The bad paragraph starts with "El algoritmo DP reduce el costo computacional en 63.58%"
    # Paragraph 375 also mentions "63.58%" but only in the [NOTA:] annotation — skip it.
    for para in doc.paragraphs:
        text = para.text
        if "63.58%" in text and "[NOTA:" not in text:
            print(f"  Found: {text[:90]!r}")
            delete_para(para)
            print("  [OK] Deleted")
            return
    print("  [WARN] Stale paragraph not found")


def step3b_clean_nota(doc):
    """Remove [NOTA: ...] annotation from end of the valid results paragraph."""
    print("\n[3b] Cleaning [NOTA] annotation...")
    para = find_para(doc, "[NOTA:")
    if para is None:
        print("  [WARN] [NOTA:] not found")
        return
    full = para.text
    idx = full.index("[NOTA:")
    clean = full[:idx].rstrip()
    print(f"  Before: {full[:100]!r}")
    replace_para_text(para, clean)
    print(f"  After:  {clean[:100]!r}")


def step3c_fix_hypothesis(doc):
    """Fix 'tiempo de procesamiento y recursos computacionales' in hypothesis."""
    print("\n[3c] Fixing hypothesis text...")
    OLD = ("reduciendo el tiempo de procesamiento y los recursos computacionales "
           "necesarios, sin comprometer la calidad de los resultados.")
    NEW = ("reduciendo el costo computacional de inferencia, sin comprometer "
           "la calidad semántica de los resultados.")
    for para in doc.paragraphs:
        if "tiempo de procesamiento" in para.text and "recursos computacionales" in para.text:
            full = para.text
            if OLD in full:
                replace_para_text(para, full.replace(OLD, NEW))
            else:
                replace_para_text(para, full.replace(
                    "tiempo de procesamiento y los recursos computacionales necesarios",
                    "costo computacional de inferencia",
                ))
            print("  [OK] Fixed")
            return
    print("  [WARN] Not found")


def step3d_strengthen_oe1(doc):
    print("\n[3d] Strengthening OE1...")
    for para in doc.paragraphs:
        if "algoritmo de programación dinámica aplicable a la inferencia segmentada" in para.text:
            replace_para_text(para,
                "Desarrollar un algoritmo de programación dinámica que minimice la función de "
                "costo compuesta costo(i,j) = tokens(i,j)² · (1 + λ·(1−coh(i,j))) + Cf para la "
                "inferencia segmentada en LLM, derivando analíticamente el número óptimo de "
                "segmentos k* = N·√((1+λ·(1−coh_avg))/Cf).")
            print("  [OK] OE1 updated")
            return
    print("  [WARN] OE1 text not found")


def step3e_strengthen_oe2(doc):
    print("\n[3e] Strengthening OE2...")
    for para in doc.paragraphs:
        if "Implementar el algoritmo de programación dinámica en un entorno de desarrollo" in para.text:
            replace_para_text(para,
                "Implementar el algoritmo de programación dinámica con garantía de optimalidad "
                "global en un sistema modular de 6 módulos (173 pruebas unitarias), con interfaz "
                "Streamlit para experimentación interactiva, utilizando Python, sentence-transformers, "
                "tiktoken y spaCy.")
            print("  [OK] OE2 updated")
            return
    print("  [WARN] OE2 text not found")


def step3f_add_boxplot_note(doc):
    """Insert explanatory note after main results paragraph (before boxplot figure)."""
    print("\n[3f] Adding boxplot clarification note...")
    anchor = find_para(doc, "corrección FDR (α=0.05).")
    if anchor:
        insert_para_after(doc, anchor,
            "La Figura 6 muestra las distribuciones de reducción de costo para cada método. "
            "El eje y=0 corresponde al DP Óptimo como referencia; valores negativos indican "
            "un costo mayor que DP. Nótese que DP+Overlap aparece en −0.30%, indicando "
            "equivalencia práctica con DP 1D bajo los parámetros canónicos (μ=0.03).")
        print("  [OK] Note inserted")
    else:
        print("  [WARN] FDR anchor not found")


def step3g_add_coherence_note(doc):
    """Insert coherence metric clarification after Pk boundary metrics paragraph."""
    print("\n[3g] Adding coherence metric clarification...")
    anchor = find_para(doc, "TextTiling muestra mayor acuerdo con el DP (Pk=0.546")
    if anchor:
        insert_para_after(doc, anchor,
            "Nota metodológica: la coherencia semántica intra-segmento (0.8994 DP vs 0.4960 "
            "Baseline) mide la similitud coseno promedio entre oraciones dentro de un mismo "
            "segmento — una métrica de cohesión interna. Las métricas Pk y WindowDiff, así "
            "como la similitud de frontera mostrada en los estudios de caso (0.35–0.53), miden "
            "el acuerdo entre fronteras de distintos métodos — una métrica de alineación externa. "
            "Ambas dimensiones son complementarias y reflejan aspectos distintos de la calidad "
            "de segmentación.")
        print("  [OK] Coherence note inserted")
    else:
        print("  [WARN] Pk anchor not found")


# ── Step 4: Figure cleanup ─────────────────────────────────────────────────────

# Old figure index entries (in REFERENCIAS section) - delete before inserting inline figures
OLD_FIGURE_INDEX = [
    "Figura 1. Distribución de tokens y coherencia semántica del corpus de evaluación",
    "Figura 2. Análisis de sensibilidad: número óptimo de segmentos k* y reducción de costo en función de λ × Cf",
    "Figura 3. Boxplots de reducción de costo por método. DP vs. Baseline: reducción media 43.88%",
    "Figura 4. Complejidad computacional empírica: tiempo de ejecución",
    "Figura 5. Similitud coseno en fronteras de segmentación",
    "Figura 6. Comparación de tokens por segmento entre métodos",
    "Figura 7. Curva de costo vs k para el corpus",
    # §4.4 standalone text-only caption (different from index entries)
    "Figura 7. Análisis de sensibilidad: efecto de λ y Cf",
]


def step4_delete_old_figure_entries(doc):
    """Delete outdated figure index entries and §4.4 text placeholder."""
    print("\n[4-cleanup] Deleting old figure index entries and §4.4 text caption...")
    deleted = 0
    for fragment in OLD_FIGURE_INDEX:
        para = find_para(doc, fragment)
        if para:
            print(f"  Deleting: {para.text[:80]!r}")
            delete_para(para)
            deleted += 1
        else:
            print(f"  [WARN] Not found: {fragment[:70]!r}")
    print(f"  [OK] Deleted {deleted}/{len(OLD_FIGURE_INDEX)} entries")


def step4_insert_figures(doc):
    """Insert all 10 chapter figures inline at their correct section positions."""
    print("\n[4] Inserting 10 inline chapter figures...")

    figures = [
        # §3.6 — after last paragraph of the section
        (
            "La extensión dp[i][k] agrega el número de segmentos k como segunda dimensión",
            FIGS / "cap3/costo_vs_lmax.png",
            "Figura 1. Costo total de inferencia en función de lmax para cada método de segmentación. Elaboración propia.",
        ),
        # Insert Figura 2 after the Figura 1 caption just created
        (
            "Figura 1. Costo total de inferencia en función de lmax",
            FIGS / "cap3/reduccion_vs_lmax.png",
            "Figura 2. Reducción de costo del DP óptimo respecto al Baseline en función de lmax. Elaboración propia.",
        ),
        # §4.2 — after last paragraph of the section
        (
            "el módulo suggest_lambda() del sistema",
            FIGS / "cap4/corpus_tokens_por_dominio.png",
            "Figura 3. Distribución de tokens por dominio en el corpus de 20 textos. Elaboración propia.",
        ),
        (
            "Figura 3. Distribución de tokens por dominio",
            FIGS / "cap4/corpus_coherencia_por_dominio.png",
            "Figura 4. Coherencia semántica promedio por dominio temático del corpus. Elaboración propia.",
        ),
        # §4.4 — after the body paragraph (old text caption was already deleted)
        (
            "no cambia el k* óptimo",
            FIGS / "cap4/sensitivity_reduccion.png",
            "Figura 5. Análisis de sensibilidad: efecto de λ y Cf sobre k* y reducción de costo DP vs Baseline. Elaboración propia.",
        ),
        # §5.1 OE3 — after the note inserted by step3f
        (
            "Figura 6 muestra las distribuciones de reducción de costo",
            FIGS / "cap5/statistical_boxplots.png",
            "Figura 6. Distribuciones de reducción de costo por método (n=20 textos). Elaboración propia.",
        ),
        # §5.1 OE4 — 3 case studies after OE4 heading
        (
            "OE4 — Evaluación de precisión: coherencia semántica y métricas de frontera",
            FIGS / "cap5/case_studies/case_llm_inference.png",
            "Figura 7. Estudio de caso: llm_inference — segmentación DP vs Baseline con métricas de frontera. Elaboración propia.",
        ),
        (
            "Figura 7. Estudio de caso: llm_inference",
            FIGS / "cap5/case_studies/case_transformer_architecture.png",
            "Figura 8. Estudio de caso: transformer_architecture — segmentación DP vs Baseline. Elaboración propia.",
        ),
        (
            "Figura 8. Estudio de caso: transformer_architecture",
            FIGS / "cap5/case_studies/case_text_segmentation.png",
            "Figura 9. Estudio de caso: text_segmentation — segmentación DP vs Baseline. Elaboración propia.",
        ),
        # §5.1 OE5 — after OE5 heading
        (
            "OE5 — Análisis de ventajas y limitaciones del enfoque propuesto",
            FIGS / "cap5/complexity_loglog.png",
            "Figura 10. Complejidad computacional del algoritmo DP en escala log-log. Elaboración propia.",
        ),
    ]

    for anchor_text, img_path, caption in figures:
        if not img_path.exists():
            print(f"  [ERROR] Missing: {img_path}")
            continue
        insert_figure_after(doc, anchor_text, str(img_path), caption)


def step4_add_annex_c(doc):
    """Append Annex C with 3 MAIN_ figures at end of document."""
    print("\n[4-annexC] Adding Annex C (3 main figures)...")

    doc.add_heading(
        "ANEXO C — DISTRIBUCIÓN DE TOKENS POR DOMINIO (CORPUS COMPLETO)", level=1
    )
    doc.add_paragraph(
        "Las siguientes figuras muestran la distribución de tokens por segmento para los tres "
        "textos principales del corpus. Las 17 figuras adicionales del corpus completo están "
        "disponibles en outputs/figures/anexo/ del repositorio del proyecto."
    )

    annex_figs = [
        (
            FIGS / "anexo/MAIN_tokens_llm_inference.png",
            "Figura 11. Distribución de tokens por segmento — llm_inference. Elaboración propia.",
        ),
        (
            FIGS / "anexo/MAIN_tokens_constitutional_law.png",
            "Figura 12. Distribución de tokens por segmento — constitutional_law. Elaboración propia.",
        ),
        (
            FIGS / "anexo/MAIN_tokens_amazon_river_basin.png",
            "Figura 13. Distribución de tokens por segmento — amazon_river_basin. Elaboración propia.",
        ),
    ]

    for img_path, caption in annex_figs:
        if not img_path.exists():
            print(f"  [ERROR] Missing: {img_path}")
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(img_path), width=Inches(5.5))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in cap.runs:
            r.italic = True
            r.font.size = Pt(10)
        print(f"  [OK] {img_path.name}")


# ── Diagnostics ────────────────────────────────────────────────────────────────

def scan_figure_refs(doc):
    print("\n[SCAN] Paragraphs containing 'Figura':")
    for para in doc.paragraphs:
        if re.search(r"Figura\s+\d+", para.text):
            print(f"  {para.text[:110]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Thesis update script ===\n")

    if not DOCX.exists():
        print(f"[ERROR] Not found: {DOCX}")
        return

    print(f"Backing up → {BACKUP.name}")
    shutil.copy2(DOCX, BACKUP)

    doc = Document(str(DOCX))
    scan_figure_refs(doc)

    # Text changes (order matters: 3a before 3b to avoid finding wrong para)
    step3a_delete_stale_para(doc)
    step3b_clean_nota(doc)
    step3c_fix_hypothesis(doc)
    step3d_strengthen_oe1(doc)
    step3e_strengthen_oe2(doc)
    step3f_add_boxplot_note(doc)
    step3g_add_coherence_note(doc)

    # Figure work (delete old entries first, then insert inline)
    step4_delete_old_figure_entries(doc)
    step4_insert_figures(doc)
    step4_add_annex_c(doc)

    doc.save(str(DOCX))
    print(f"\n=== Done — saved {DOCX.name} ===")


if __name__ == "__main__":
    main()
