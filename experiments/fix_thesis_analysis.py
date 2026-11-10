#!/usr/bin/env python3
"""
Correcciones del análisis profundo — 13 ítems (C1-C4, I1-I5, M1-M4).
Enfoque de tesis: lenguaje académico formal en español, coherencia interna.
"""
from docx import Document
from lxml import etree
import copy

DOC = "docs/thesis/Tesis_Churata_Apaza_2025_APA6_FINAL.docx"
W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"


# ── helpers ──────────────────────────────────────────────────────────────────

def get_text(elem):
    return "".join(t.text or "" for t in elem.iter(f"{{{W}}}t"))

def set_text(elem, new_text):
    for r in list(elem.findall(f"{{{W}}}r")):
        elem.remove(r)
    r = etree.SubElement(elem, f"{{{W}}}r")
    t = etree.SubElement(r, f"{{{W}}}t")
    t.text = new_text
    t.set(f"{{{XML}}}space", "preserve")

def replace_in_para(elem, old, new):
    txt = get_text(elem)
    if old not in txt:
        return False
    set_text(elem, txt.replace(old, new))
    return True

def insert_clone_after(anchor_elem, new_text):
    """Clona anchor_elem, reemplaza su texto y lo inserta inmediatamente después."""
    clone = copy.deepcopy(anchor_elem)
    set_text(clone, new_text)
    anchor_elem.addnext(clone)
    return clone


# ── cargar ────────────────────────────────────────────────────────────────────

doc   = Document(DOC)
paras = doc.paragraphs
e     = [p._element for p in paras]   # refs estables

# ═══════════════════════════════════════════════════════════════════════════════
# C1 — §1.1: "reduciendo el tiempo de procesamiento" → "costo computacional"
# idx 184
# ═══════════════════════════════════════════════════════════════════════════════
ok = replace_in_para(e[184],
    "reduciendo el tiempo de procesamiento",
    "reduciendo el costo computacional de inferencia"
)
print(f"✓ C1 — §1.1 actualizado: {ok}")

# ═══════════════════════════════════════════════════════════════════════════════
# C2 — §3.3.1 Código 3: corregir llamada a _compute_cost y agregar prefix_sims
#   [338] embeddings = ...  →  agregar línea prefix_sims después
#   [344] # O(n^2)         →  # O(n·w), w acotado por lmax/token_avg
#   [349] embeddings       →  prefix_sims
# ═══════════════════════════════════════════════════════════════════════════════
# Insertar nueva línea de código después de [338]
insert_clone_after(e[338], "    _, prefix_sims = _build_similarity_arrays(embeddings)")

# Ahora los índices originales [344] y [349] siguen siendo accesibles por ref
ok1 = replace_in_para(e[344], "# O(n^2)", "# O(n·w), w acotado por lmax/token_avg")
ok2 = replace_in_para(e[349], "_compute_cost(span, embeddings, i, j,",
                               "_compute_cost(span, prefix_sims, i, j,")
print(f"✓ C2 — §3.3.1 Código 3 corregido (prefix_sims={ok2}, comentario={ok1})")

# ═══════════════════════════════════════════════════════════════════════════════
# C3 — §2.3.3: O(n²) → O(n·w) con explicación correcta
# idx 266
# ═══════════════════════════════════════════════════════════════════════════════
set_text(e[266], (
    "El problema de segmentación pertenece a la clase de problemas de partición "
    "óptima de secuencias. La complejidad temporal del algoritmo DP es O(n·w), "
    "donde n es el número de oraciones y w = lmax/token_avg es el tamaño de la "
    "ventana de búsqueda acotada por las restricciones de tokens mediante bisect; "
    "en el caso peor sin restricciones de tokens, la complejidad degrada a O(n²). "
    "La complejidad espacial es O(n). "
    "La reconstrucción mediante backtracking tiene complejidad O(k), donde k es "
    "el número de segmentos en la solución óptima."
))
print("✓ C3 — §2.3.3 complejidad corregida a O(n·w)")

# ═══════════════════════════════════════════════════════════════════════════════
# C4 — §5.2: "reducción de recursos computacionales" → "reducción del costo
#             computacional de inferencia"
# idx 543
# ═══════════════════════════════════════════════════════════════════════════════
ok = replace_in_para(e[543],
    "Primero, reducción de recursos computacionales:",
    "Primero, reducción del costo computacional de inferencia:"
)
print(f"✓ C4 — §5.2 verificación alineada con hipótesis: {ok}")

# ═══════════════════════════════════════════════════════════════════════════════
# I1 — §1.4: Pregunta 3 — acotar a lo que los experimentos realmente miden
# idx 197
# ═══════════════════════════════════════════════════════════════════════════════
set_text(e[197], (
    "¿En qué medida la solución de programación dinámica reduce el costo "
    "computacional de inferencia (tokens²) y preserva la coherencia semántica "
    "intra-segmento respecto a los enfoques de segmentación estática (Baseline "
    "greedy, Sliding Window) y heurística (TextTiling)?"
))
print("✓ I1 — §1.4 Pregunta 3 acotada a métricas efectivamente medidas")

# ═══════════════════════════════════════════════════════════════════════════════
# I2 — §4.2: lista de dominios reorganizada para que coincida con los 15
#             categorías reales del análisis de corpus
# idx 484
# ═══════════════════════════════════════════════════════════════════════════════
old_i2 = get_text(e[484])
new_i2 = old_i2.replace(
    "cubriendo 15 dominios temáticos con diversidad controlada de longitud y "
    "complejidad semántica. El corpus fue generado específicamente para este "
    "trabajo dado que no existe un corpus estándar en español para evaluación de "
    "segmentación de textos técnicos orientados a LLM.",
    "cubriendo 15 dominios temáticos con diversidad controlada de longitud y "
    "complejidad semántica: LLM/Inferencia (2 textos), Algoritmos/Programación "
    "Dinámica (2), Aprendizaje Profundo/Transformers (2), Procesamiento de "
    "Lenguaje Natural (1), Aprendizaje Automático (1), Recuperación de "
    "Información (1), Ingeniería de Software (1), Historia (2), "
    "Biología/Genética (2), Medicina/Fisiología (1), Física/Mecánica Cuántica "
    "(1), Economía (1), Filosofía (1), Derecho (1) y Geografía (1). "
    "El corpus fue generado específicamente para este trabajo dado que no existe "
    "un corpus estándar en español para evaluación de segmentación de textos "
    "técnicos orientados a LLM."
)
if new_i2 != old_i2:
    set_text(e[484], new_i2)
    print("✓ I2 — §4.2 lista de 15 dominios corregida y consistente con corpus_analysis.json")
else:
    print("⚠ I2 — texto no encontrado, revisar manualmente")

# ═══════════════════════════════════════════════════════════════════════════════
# I3 — §4.4: eliminar "invarianza de escala" duplicada del párrafo [496]
# Se mantiene solo la primera aparición (en [494]); se borra del segundo párrafo.
# idx 496
# ═══════════════════════════════════════════════════════════════════════════════
old_i3 = get_text(e[496])
# Quitar la frase duplicada del final del párrafo
dup = (" El sistema exhibe invarianza de escala: multiplicar λ por una constante "
       "c y dividir Cf por c² no cambia el k* óptimo.")
ok = replace_in_para(e[496], dup, "")
print(f"✓ I3 — §4.4 'invarianza de escala' duplicada eliminada: {ok}")

# ═══════════════════════════════════════════════════════════════════════════════
# I4 — Anexo B: eliminar referencias duplicadas al final (idx 960-963)
# Estructura: [957-959] primera copia, [960] vacío, [961-963] duplicados
# ═══════════════════════════════════════════════════════════════════════════════
for idx in [960, 961, 962, 963]:
    e[idx].getparent().remove(e[idx])
print("✓ I4 — Anexo B: 4 entradas duplicadas (Hearst, Beeferman, Pevzner + vacío) eliminadas")

# ═══════════════════════════════════════════════════════════════════════════════
# I5 — §1.6.1 OE4: descripción genérica → métricas concretas
# idx 206
# ═══════════════════════════════════════════════════════════════════════════════
set_text(e[206], (
    "Evaluar la precisión de la inferencia segmentada del algoritmo propuesto "
    "mediante coherencia semántica intra-segmento (similitud coseno promedio entre "
    "oraciones adyacentes), métricas de frontera Pk (Beeferman et al., 1999) y "
    "WindowDiff (Pevzner y Hearst, 2002), comparando cuantitativamente con los "
    "métodos Baseline greedy, Sliding Window y TextTiling sobre el mismo corpus."
))
print("✓ I5 — §1.6.1 OE4 ampliado con métricas concretas (Pk, WindowDiff, coherencia coseno)")

# ═══════════════════════════════════════════════════════════════════════════════
# M1 — §2.2.3: índice k en la fórmula de coherencia → ℓ (evita colisión con
#              k = número de segmentos usado en todo el documento)
# idx 254
# ═══════════════════════════════════════════════════════════════════════════════
set_text(e[254], (
    "coherencia(i, j) = [1 / (j−i−1)] · Σ cosine_similarity(eℓ, eℓ+1),"
    "    ℓ = i,...,j−2"
))
print("✓ M1 — §2.2.3 índice de suma cambiado k→ℓ para evitar colisión de notación")

# ═══════════════════════════════════════════════════════════════════════════════
# M2 — §2.4.3: sección de métricas vacía → párrafo descriptivo antes de Tabla 2
# idx 274  (actualmente: "Las métricas ... se presentan en la Tabla 2.")
# ═══════════════════════════════════════════════════════════════════════════════
set_text(e[274], (
    "Las métricas de evaluación se organizan en tres dimensiones. "
    "(1) Eficiencia computacional: costo de inferencia acumulado en tokens², "
    "definido como Σ tokens(i,j)² sobre todos los segmentos; captura directamente "
    "el costo cuadrático de la autoatención. "
    "(2) Coherencia semántica intra-segmento: media de la similitud coseno entre "
    "embeddings de oraciones adyacentes dentro de cada segmento; rango [−1, 1], "
    "valores próximos a 1 indican alta homogeneidad temática. "
    "(3) Precisión de fronteras: Pk (Beeferman et al., 1999) mide la probabilidad "
    "de error en una ventana deslizante de tamaño k; WindowDiff (Pevzner y Hearst, "
    "2002) penaliza adicionalmente el exceso de fronteras detectadas. "
    "Los valores, rangos y referencias de cada métrica se resumen en la Tabla 2."
))
print("✓ M2 — §2.4.3 métricas descritas en prosa (tokens², coherencia, Pk, WindowDiff)")

# ═══════════════════════════════════════════════════════════════════════════════
# M3 — §3.3.1: nota sobre defaults lmin=50/lmax=200 vs parámetros canónicos
#              Insertar como párrafo Normal después del label "Código 3." (idx 359)
# ═══════════════════════════════════════════════════════════════════════════════
insert_clone_after(e[359], (
    "Nota. Los valores por defecto de segment_dp (lmin=50, lmax=200) corresponden "
    "a la interfaz pública de la función. Los experimentos del Capítulo IV emplean "
    "los parámetros canónicos lmin=20, lmax=150, especificados explícitamente en "
    "cada llamada (véase §4.1)."
))
print("✓ M3 — §3.3.1 nota agregada sobre defaults vs parámetros canónicos")

# ═══════════════════════════════════════════════════════════════════════════════
# M4 — §3.6: expandir de un párrafo escueto a descripción completa del DP 2D
#             con recursión, complejidad y relación con k* analítico
# idx 401
# ═══════════════════════════════════════════════════════════════════════════════
set_text(e[401], (
    "La extensión bidimensional dp[j][k] calcula el costo mínimo de segmentar las "
    "primeras j oraciones en exactamente k segmentos para todos los valores de k "
    "simultáneamente, mediante la recursión de Bellman bidimensional: "
    "dp[0][0] = 0; dp[j][k] = min{ dp[i][k−1] + costo(i,j) } para "
    "lmin ≤ tokens(i,j) ≤ lmax. "
    "El resultado es la curva costo(k) = dp[n][k], de forma en U, que identifica "
    "el número óptimo de segmentos k* = argmin_k {dp[n][k]} sin búsqueda "
    "exhaustiva adicional. La complejidad temporal es O(n²·k_max) y la espacial "
    "O(n·k_max). La fórmula analítica k* = N·√((1+λ·(1−coh_avg))/Cf) —derivada "
    "en §4.4— aproxima el mínimo de esta curva en tiempo O(1), evitando la "
    "ejecución completa del DP 2D cuando solo se requiere estimar k*. La Figura 1 "
    "muestra la curva costo vs k para un texto representativo del corpus, con el "
    "mínimo de la curva en U identificando k*."
))
print("✓ M4 — §3.6 DP 2D expandido con recursión, complejidad y relación con k* analítico")

# ── guardar ──────────────────────────────────────────────────────────────────
doc.save(DOC)
print(f"\nDocumento guardado: {DOC}")
print("\nNota C5 (pendiente manual): verificar en Google Scholar las 3 referencias")
print("  marcadas '[Verificar datos antes de la defensa]' —")
print("  Kitaev & Klein (2023), Li et al. (2024), Wang & Liu (2023).")
