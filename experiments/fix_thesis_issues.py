#!/usr/bin/env python3
"""
Correcciones tesis (9 ítems):
  1. §2.4.1 — elimina nota CORRECCIÓN, inserta texto limpio
  2. §3.3.2 — código roto → código correcto de dp.py
  3. §4.1.1 — OEs en orden inverso (OE5→OE1) → OE1→OE5
  4. §4.1.1 — Discusión OE1 al lugar equivocado → moverla a §5.1
  5. §5.2   — hipótesis con redacción antigua → redacción corregida
  6. §5.1   — Figuras 5 y 6 fuera de lugar y mal numeradas
  7. §4.4   — "n=108 tokens" confuso → "n=108 oraciones"
  8. §4.1.1 — Grid OE5 incorrecto → valores reales del script
  9. §4.1   — numeración mixta (i),(ii),(3)... → (1),(2),(3)...
"""
from docx import Document
from lxml import etree

DOC = "docs/thesis/Tesis_Churata_Apaza_2025_APA6_FINAL.docx"
W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"


# ── helpers ──────────────────────────────────────────────────────────────────

def get_text(elem: etree._Element) -> str:
    return "".join(t.text or "" for t in elem.iter(f"{{{W}}}t"))


def set_text(elem: etree._Element, new_text: str) -> None:
    """Replace all runs in a paragraph element with a single run of new_text.
    Paragraph properties (pPr) and non-run children (bookmarks, etc.) are kept."""
    for r in list(elem.findall(f"{{{W}}}r")):
        elem.remove(r)
    r = etree.SubElement(elem, f"{{{W}}}r")
    t = etree.SubElement(r, f"{{{W}}}t")
    t.text = new_text
    t.set(f"{{{XML}}}space", "preserve")


def replace_in_para(elem: etree._Element, old: str, new: str) -> bool:
    txt = get_text(elem)
    if old not in txt:
        return False
    set_text(elem, txt.replace(old, new))
    return True


# ── load ─────────────────────────────────────────────────────────────────────

doc   = Document(DOC)
paras = doc.paragraphs          # snapshot
e     = [p._element for p in paras]   # element refs (stable across mutations)

# ── Issue 1: §2.4.1 ─ reemplazar nota CORRECCIÓN ────────────────────────────
# idx 270: único párrafo del apartado, contiene "CORRECCIÓN: ..."
set_text(e[270], (
    "La función de costo compuesta modela el costo de procesar el segmento S(i,j) "
    "mediante el producto del costo cuadrático de la autoatención por un factor de "
    "coherencia semántica, más un costo fijo por llamada al LLM: "
    "costo(i,j) = tokens(i,j)² · [1 + λ · (1 − coherencia(i,j))] + Cf, "
    "donde tokens(i,j) es el número de tokens del segmento, coherencia(i,j) es la "
    "similitud coseno promedio entre oraciones adyacentes del segmento, λ ≥ 0 es el "
    "peso de coherencia semántica, y Cf > 0 es el costo fijo por invocación al LLM. "
    "La estructura multiplicativa garantiza que λ tenga peso proporcional al costo "
    "cuadrático dominante: con t² ≈ 10.000, un enfoque aditivo puro haría que el "
    "término de coherencia aportara menos del 0,001 % del costo total."
))
print("✓ Issue 1 — §2.4.1 CORRECCIÓN reemplazada")

# ── Issue 2: §3.3.2 ─ código roto (idx 364-377) → código correcto ───────────
# Código correcto de dp.py (6 líneas); las 8 líneas sobrantes se eliminan.
correct_lines = [
    "def _compute_cost(token_count, prefix_sims, start, end,",
    "                  coherence_lambda, fixed_cost):",
    "    t2 = float(token_count ** 2)",
    "    avg_coh = _avg_coherence(prefix_sims, start, end)",
    "    coherence_factor = 1.0 + coherence_lambda * (1.0 - avg_coh)",
    "    return t2 * coherence_factor + fixed_cost",
]
for i, line in enumerate(correct_lines):
    set_text(e[364 + i], line)
for idx in range(370, 378):      # eliminar 8 líneas sobrantes del código roto
    e[idx].getparent().remove(e[idx])
print("✓ Issue 2 — §3.3.2 código corregido (6 líneas limpias, 8 sobrantes eliminadas)")

# ── Issue 5: §5.2 ─ hipótesis con redacción antigua ─────────────────────────
ok = replace_in_para(e[550],
    "reduciendo el tiempo de procesamiento y los recursos computacionales "
    "necesarios, sin comprometer la calidad de los resultados",
    "reduciendo el costo computacional de inferencia, sin comprometer la "
    "calidad de los resultados",
)
print(f"✓ Issue 5 — §5.2 hipótesis corregida: {ok}")

# ── Issue 7: §4.4 ─ "n=108 tokens" → "n=108 oraciones" ─────────────────────
ok = replace_in_para(e[505], "n=108 tokens", "n=108 oraciones")
print(f"✓ Issue 7 — §4.4 n=108 oraciones: {ok}")

# ── Issue 8: §4.1.1 OE5 Fase 2 ─ grid incorrecto ───────────────────────────
ok = replace_in_para(e[452],
    "λ ∈ {0.1, 0.3, 0.5, 1.0, 2.0} y Cf ∈ {500, 1000, 2000, 3000, 5000}",
    "λ ∈ {0.0, 0.5, 1.0, 1.5, 2.0} y Cf ∈ {200, 500, 1000, 2000, 5000}",
)
print(f"✓ Issue 8 — §4.1.1 OE5 grid corregido: {ok}")

# ── Issue 9: §4.1 ─ numeración mixta (i),(ii),(3)... → (1),(2),(3)... ────────
ok = replace_in_para(e[447],
    "(i) DP Óptimo 1D, (ii) DP 2D (k* analítico), (3)",
    "(1) DP Óptimo 1D, (2) DP 2D (k* analítico), (3)",
)
print(f"✓ Issue 9 — §4.1 numeración unificada: {ok}")

# ── Issue 6: §5.1 ─ figuras desplazadas y mal numeradas ─────────────────────
# Estructura actual al inicio de §5.1:
#   e[511] vacío | e[512] imagen (boxplot) | e[513] "Figura 5. Boxplots..."
#   e[514] vacío | e[515] imagen (log-log) | e[516] "Figura 6. Complejidad..."
# Objetivo:
#   grupo {511,512,513} → renombrar "Figura 5" → "Figura 6" + mover tras e[528]
#   grupo {514,515,516} → renombrar "Figura 6" → "Figura 10" + mover antes e[547]
replace_in_para(e[513], "Figura 5.", "Figura 6.")
replace_in_para(e[516], "Figura 6.", "Figura 10.")

# Mover grupo boxplot (Fig6) tras "La Figura 6 muestra..." (e[528], en OE3)
anchor = e[528]
prev   = anchor
for elem in [e[511], e[512], e[513]]:
    prev.addnext(elem)
    prev = elem

# Mover grupo log-log (Fig10) antes Discusión OE5 (e[547])
disc_oe5 = e[547]
for elem in [e[514], e[515], e[516]]:
    disc_oe5.addprevious(elem)

print("✓ Issue 6 — §5.1 figuras renombradas y reubicadas")

# ── Issues 3+4: §4.1.1 ─ reordenar OEs + mover Discusión OE1 ───────────────
# Mapa de bloques (cabecera OE + 6 fases, 7 párrafos cada uno):
#   e[450-456] OE5 | e[457-463] OE4 | e[464-470] OE3
#   e[471-477] OE2 | e[478-484] OE1 | e[485] Discusión OE1
#
# Insertar tras el párrafo intro e[449] en orden OE1→OE2→OE3→OE4→OE5.
# lxml mueve el elemento de su posición actual al nuevo lugar.

blocks = {
    "OE1": [e[i] for i in range(478, 485)],
    "OE2": [e[i] for i in range(471, 478)],
    "OE3": [e[i] for i in range(464, 471)],
    "OE4": [e[i] for i in range(457, 464)],
    "OE5": [e[i] for i in range(450, 457)],
}
disc_oe1 = e[485]

prev = e[449]   # "A continuación se describe..."
for name, block in blocks.items():
    for elem in block:
        prev.addnext(elem)
        prev = elem

# Mover Discusión OE1 a §5.1 tras el último párrafo de OE1 (e[520])
e[520].addnext(disc_oe1)

print("✓ Issues 3+4 — §4.1.1 OEs en orden OE1→OE5; Discusión OE1 movida a §5.1")

# ── guardar ──────────────────────────────────────────────────────────────────
doc.save(DOC)
print(f"\nDocumento guardado: {DOC}")
