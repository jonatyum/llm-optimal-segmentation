import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.segmentation.dp import segment_dp, segment_dp_2d
from src.segmentation.baseline import segment_baseline
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.texttiling import segment_texttiling
from src.segmentation.metrics import compute_metrics, compare
from src.segmentation.evaluator import evaluate_segmentation
from src.segmentation.calibration import suggest_lambda, estimate_optimal_k, validate_calibration
from src.segmentation.segmentation_metrics import boundaries_from_result, pk, window_diff

st.set_page_config(
    page_title="Segmentación Óptima para LLM",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── BASE ───────────────────────────────────────────────────────────────── */
/* Fuente base más grande para visibilidad en proyección (todo lo que usa rem escala). */
html { font-size: 18.5px; }
.main .block-container { padding-top: 1rem; padding-bottom: 3rem; }

/* Texto de párrafos, listas, sliders y selectores un poco más grande. */
.main p, .main li,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, .stRadio label, .stCheckbox label { font-size: 1.03rem; }
[data-testid="stTabs"] button { font-size: 1.02rem !important; }
[data-testid="stExpander"] summary p { font-size: 1.0rem; }

/* ── CAPTIONS ────────────────────────────────────────────────────────────── */
/* Usa el color de texto del tema con opacidad: legible tanto en claro como en oscuro. */
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] { color: var(--text-color) !important; opacity: 0.68; font-size: 0.82rem !important; }

/* ── METRIC CARDS ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(59,130,246,0.14);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}

/* ── BOTONES ─────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    padding: 0.55rem 2rem;
    box-shadow: 0 2px 10px rgba(59,130,246,0.40);
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(59,130,246,0.50);
}
.stButton > button[kind="secondary"] {
    border: 1.5px solid #cbd5e1;
    color: #334155 !important;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #3b82f6;
    color: #1d4ed8 !important;
    background: #eff6ff !important;
}

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span { color: #e2e8f0 !important; }

/* ── TABS ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button { color: #64748b !important; font-weight: 500; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1e40af !important;
    font-weight: 700 !important;
    border-bottom: 2.5px solid #3b82f6 !important;
}

/* ── EXPANDERS ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary {
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 7px 12px;
    color: #1e3a5f !important;
    font-weight: 500;
}
[data-testid="stExpander"] summary:hover { background: #eff6ff; }

/* ── PROGRESS ────────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    border-radius: 99px;
}

/* ── ALERTS ──────────────────────────────────────────────────────────────── */
/* No forzar un color oscuro fijo: en modo oscuro deja el texto invisible.
   Streamlit ya da contraste a sus alertas; solo reforzamos el peso. */
[data-testid="stAlert"] p { font-weight: 500; }

/* ── TEXT AREA ───────────────────────────────────────────────────────────── */
textarea { border-radius: 8px !important; font-size: 0.88rem !important; }

/* ── DATAFRAME ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 26px 32px 22px;
    margin-bottom: 20px;
">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px">
        <span style="font-size:2rem">✂️</span>
        <h1 style="margin:0; color:#f8fafc; font-size:1.7rem; font-weight:800; letter-spacing:-0.4px">
            Segmentación Óptima para LLM
        </h1>
    </div>
    <p style="margin:0; color:#94a3b8; font-size:0.88rem; padding-left:52px">
        Comparación visual de métodos de segmentación · UMSA 2026
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parámetros")
    st.caption("Ajusta los parámetros para explorar el comportamiento del algoritmo")

    lmin = st.slider(
        "Lmin — tokens mínimos por segmento",
        5, 100, 20, step=5,
        help="Límite inferior de tokens por segmento. Evita segmentos demasiado pequeños que pierden contexto semántico."
    )
    lmax = st.slider(
        "Lmax — tokens máximos por segmento",
        50, 400, 150, step=10,
        help="Límite superior de tokens por segmento. Representa la ventana de contexto disponible del LLM. "
             "Valor canónico de la tesis: 150."
    )
    overlap = st.slider(
        "Solapamiento — ventana deslizante",
        0, 100, 20, step=5,
        help="Tokens de solapamiento entre ventanas consecutivas en el método Sliding Window tradicional."
    )
    coherence_lambda = st.slider(
        "λ — peso de coherencia semántica",
        0.0, 2.0, 0.5, step=0.1,
        help="Controla cuánto penaliza el algoritmo los cortes que rompen unidades semánticas. λ=0 ignora coherencia, λ=2 la prioriza sobre eficiencia."
    )
    overlap_mu = st.slider(
        "μ — peso del costo de overlap",
        0.0, 2.0, 0.03, step=0.01,
        help="Penaliza el overlap excesivo entre segmentos consecutivos. μ=0 ignora el overlap, valores altos lo minimizan."
    )
    fixed_cost = st.slider(
        "Costo fijo por segmento",
        0.0, 3000.0, 2000.0, step=50.0,
        help="Overhead fijo por cada llamada al LLM. Valores altos favorecen menos segmentos más grandes. "
             "Produce la curva en U en la DP 2D. Valor canónico de la tesis: 2000."
    )
    model = st.selectbox(
        "Modelo de tokenización",
        ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        help="Modelo cuyo tokenizador se usa para contar tokens. Afecta la distribución de tokens por segmento."
    )

    st.divider()
    st.caption("**TextTiling**")
    tt_window_size = st.slider(
        "Tamaño de ventana — bloques",
        1, 10, 2, step=1,
        help="Número de oraciones a cada lado del punto de corte candidato para calcular similitud de bloque. Valores mayores suavizan la señal."
    )
    tt_smoothing = st.slider(
        "Suavizado — número de pasadas",
        0, 5, 1, step=1,
        help="Iteraciones de media móvil sobre la curva de similitud. 0 = sin suavizado."
    )

    st.divider()
    st.caption("**DP + Overlap**")
    max_overlap = st.slider(
        "Solapamiento máx. — oraciones",
        0, 10, 3, step=1,
        help="Número máximo de oraciones que pueden repetirse al inicio de cada segmento como contexto del anterior."
    )

    st.divider()
    st.caption("**Rendimiento**")
    max_sentences = st.slider(
        "Máx. oraciones (fusión automática)",
        min_value=50, max_value=1000, value=300, step=50,
        help=(
            "Limita el número de oraciones que procesa el DP. "
            "Si el texto tiene más, oraciones consecutivas se fusionan automáticamente. "
            "Reduce para textos muy largos; aumenta para mayor precisión."
        )
    )

    st.divider()
    st.caption("**Guía rápida**")
    st.caption("↑ lmax → más tokens por segmento")
    st.caption("↑ λ → más coherencia semántica")
    st.caption("↑ costo fijo → curva en U más pronunciada")

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Texto de entrada")

_DEFAULT_TEXT = (
    "Los modelos de lenguaje de gran escala han revolucionado el procesamiento automático "
    "del lenguaje natural durante la última década. Su arquitectura basada en transformadores "
    "les permite capturar dependencias de largo alcance en el texto y generar respuestas "
    "coherentes y precisas. Sin embargo, su capacidad de procesamiento está limitada por una "
    "ventana de contexto finita, lo que impone restricciones importantes al procesar documentos extensos. "
    "El costo computacional de la autoatención crece de forma cuadrática con la longitud del contexto, "
    "lo que hace que procesar documentos muy largos sea prohibitivamente costoso.\n\n"
    "La recuperación de información y el razonamiento sobre documentos largos constituyen uno de los "
    "principales desafíos en la aplicación práctica de estos modelos. Cuando un documento supera la "
    "ventana de contexto disponible, es necesario dividirlo en fragmentos manejables antes de procesarlo. "
    "La calidad de esta partición determina en gran medida la coherencia y precisión de las respuestas "
    "generadas por el modelo. Un enfoque ingenuo de corte por tamaño fijo ignora las fronteras "
    "semánticas del texto y puede romper unidades de significado completas.\n\n"
    "La segmentación óptima busca encontrar la división del texto que minimiza el costo computacional "
    "de inferencia preservando al mismo tiempo la cohesión semántica de cada fragmento. "
    "Un corte inadecuado que interrumpe una unidad temática degrada la calidad del contexto disponible "
    "para el modelo. Por el contrario, segmentos bien delimitados permiten al modelo acceder a toda "
    "la información relevante dentro de su ventana de contexto. La coherencia semántica se mide mediante "
    "la similitud coseno entre las representaciones vectoriales de oraciones consecutivas.\n\n"
    "El algoritmo de programación dinámica ofrece una solución exacta a este problema de optimización. "
    "A diferencia de los enfoques heurísticos como la ventana deslizante o el corte por tamaño fijo, "
    "la programación dinámica garantiza encontrar la partición de costo mínimo entre todas las posibles. "
    "La función de costo integra el costo cuadrático de autoatención y una penalización por incoherencia "
    "semántica entre oraciones consecutivas dentro de un mismo segmento. "
    "La complejidad del algoritmo es cuadrática en el número de oraciones, lo que lo hace tratable "
    "para documentos de tamaño moderado sin sacrificar la optimalidad de la solución."
)

if "text_widget" not in st.session_state:
    st.session_state["text_widget"] = _DEFAULT_TEXT

# Cargar corpus para el selector
_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus.json")
with open(_CORPUS_PATH, encoding="utf-8") as _f:
    _CORPUS = json.load(_f)
# Traducción de los dominios del corpus al español para la interfaz (los datos no cambian).
_DOMINIO_ES = {
    "LLM": "LLM",
    "Algorithms": "Algoritmos",
    "NLP": "PLN",
    "Deep Learning": "Aprendizaje Profundo",
    "IR": "Recuperación de Información",
    "Software Engineering": "Ingeniería de Software",
    "Machine Learning": "Aprendizaje Automático",
    "History": "Historia",
    "Biology": "Biología",
    "Economics": "Economía",
    "Philosophy": "Filosofía",
    "Law": "Derecho",
    "Medicine": "Medicina",
    "Physics": "Física",
    "Geography": "Geografía",
}
_CORPUS_OPTIONS = ["— Texto de ejemplo —"] + [
    f"{d['title']} ({_DOMINIO_ES.get(d['domain'], d['domain'])})" for d in _CORPUS
]

# Ejemplo por defecto para la demo: el texto que mejor evidencia la mejora del DP.
# "Filosofía de la Consciencia" → 50.3 % de reducción de costo vs Baseline (la mayor del
# corpus) y la curva en U más clara (mínimo en k*=13 con repunte a la derecha).
_DEFAULT_CORPUS_ID = "philosophy_of_mind"
_DEFAULT_CORPUS_INDEX = next(
    (i + 1 for i, d in enumerate(_CORPUS) if d["id"] == _DEFAULT_CORPUS_ID), 0
)

# Costo fijo didáctico SOLO para la curva en U. Con el Cf canónico (2000) y textos cortos,
# el óptimo k* cae en el extremo del rango y la curva parece una "L" (solo descenso). Un Cf
# más alto centra el mínimo y hace visible la forma en U sin alterar la comparación DP-vs-Baseline.
_UCURVE_FIXED_COST = 6000.0

# Límite de upload: .streamlit/config.toml → maxUploadSize = 5 (MB).
# Streamlit rechaza el archivo a nivel de servidor antes de que llegue aquí.
# El check Python es una segunda barrera por si se cambia config.toml sin actualizar este código.
_MAX_FILE_MB = 5.0
_MAX_FILE_BYTES = int(_MAX_FILE_MB * 1024 * 1024)

# Límite de precisión del DP: depende del slider max_sentences (se recalcula más abajo).
# Límite de embedding: all-MiniLM-L6-v2 trunca silenciosamente cada oración a 256 tokens (~190 palabras).

# Valor inicial del selector de corpus (el mejor ejemplo). Se fija en session_state para
# poder resetearlo a "— Texto de ejemplo —" cuando el usuario sube un archivo, sin que el
# selector vuelva a sobrescribir el texto subido.
if "_corpus_selector" not in st.session_state:
    st.session_state["_corpus_selector"] = _CORPUS_OPTIONS[_DEFAULT_CORPUS_INDEX]

col_upload, col_corpus = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📄 Cargar archivo .txt", type=["txt"],
        help=(
            f"Límite duro: {_MAX_FILE_MB:.0f} MB (servidor). "
            "La precisión de los cortes depende del slider 'Máx. oraciones'."
        ),
    )
    if uploaded_file is not None:
        _raw_bytes = uploaded_file.read()
        _file_mb = len(_raw_bytes) / (1024 * 1024)
        if len(_raw_bytes) > _MAX_FILE_BYTES:
            st.error(
                f"Archivo demasiado grande: **{_file_mb:.2f} MB** (límite: {_MAX_FILE_MB:.0f} MB). "
                "Divide el documento en partes o usa el corpus de ejemplo."
            )
        else:
            new_text = _raw_bytes.decode("utf-8", errors="replace").strip()
            _approx_words = len(new_text.split())
            st.caption(
                f"📁 **{uploaded_file.name}** — {_file_mb:.3f} MB · ~{_approx_words:,} palabras"
            )
            # Cargar el archivo SOLO cuando es nuevo (firma = nombre + tamaño) y, al hacerlo,
            # resetear el selector de corpus a "— Texto de ejemplo —". Sin esto, el bloque del
            # selector (que corre después) volvía a cargar el texto de prueba y pisaba el subido.
            _upload_sig = (uploaded_file.name, len(_raw_bytes))
            if new_text and st.session_state.get("_last_upload_sig") != _upload_sig:
                st.session_state["text_widget"] = new_text
                st.session_state["_last_upload_sig"] = _upload_sig
                st.session_state["_corpus_selector"] = _CORPUS_OPTIONS[0]
                st.session_state.pop("_last_corpus", None)
                st.success("✅ Archivo cargado. Da clic en **Segmentar**.")

with col_corpus:
    selected_corpus = st.selectbox(
        "Textos del corpus (pruebas)",
        _CORPUS_OPTIONS,
        help="Selecciona uno de los 20 textos del corpus para cargarlo directamente. "
             "Por defecto carga el mejor ejemplo de la demo (mayor mejora del DP).",
        key="_corpus_selector",
    )
    if selected_corpus != _CORPUS_OPTIONS[0]:
        corpus_idx = _CORPUS_OPTIONS.index(selected_corpus) - 1
        corpus_text = _CORPUS[corpus_idx]["text"]
        if st.session_state.get("_last_corpus") != selected_corpus:
            st.session_state["text_widget"] = corpus_text
            st.session_state["_last_corpus"] = selected_corpus
        # botón de descarga del texto seleccionado
        st.download_button(
            "⬇ Descargar texto",
            data=corpus_text,
            file_name=f"{_CORPUS[corpus_idx]['id']}.txt",
            mime="text/plain",
        )

text = st.text_area("Ingresa o edita el texto a segmentar", key="text_widget", height=180)

# ── Indicador de longitud y estado ───────────────────────────────────────────
_n_chars = len(text)
_n_words = len(text.split())
_n_mb = len(text.encode("utf-8")) / (1024 * 1024)
_n_sents_raw = max(1, text.count(".") + text.count("!") + text.count("?"))
_n_sents_effective = min(_n_sents_raw, max_sentences)
_will_merge = _n_sents_raw > max_sentences
_merge_ratio = _n_sents_raw / _n_sents_effective if _will_merge else 1

# Umbrales reales: dependen del slider max_sentences.
# Supuesto: ~18 palabras/oración en texto técnico/académico.
_WORDS_PER_SENT = 18
_WARN_WORDS  = max_sentences * _WORDS_PER_SENT        # fusión comienza
_HEAVY_WORDS = max_sentences * _WORDS_PER_SENT * 5   # fusión x5 → calidad degradada
_WARN_MB     = _WARN_WORDS  * 6 / 1_048_576
_HEAVY_MB    = _HEAVY_WORDS * 6 / 1_048_576

_col_info, _col_bar = st.columns([2, 3])
with _col_info:
    if _n_words < _WARN_WORDS:
        _color = "🟢"
    elif _n_words < _HEAVY_WORDS:
        _color = "🟡"
    else:
        _color = "🔴"
    _merge_note = f" · fusión {_merge_ratio:.1f}×" if _will_merge else ""
    st.caption(
        f"{_color} {_n_mb:.3f} MB · {_n_words:,} palabras · "
        f"~{_n_sents_raw:,} oraciones → procesará {_n_sents_effective:,}{_merge_note}"
    )
    st.caption(
        f"Límites reales — upload: 5 MB · precisión DP: "
        f"≤ {_WARN_MB:.2f} MB / {_WARN_WORDS:,} pal · "
        f"degradación: > {_HEAVY_MB:.2f} MB / {_HEAVY_WORDS:,} pal"
    )
with _col_bar:
    st.progress(min(_n_words / _HEAVY_WORDS, 1.0))

if _n_words > _HEAVY_WORDS:
    st.warning(
        f"Texto muy largo ({_n_words:,} palabras · {_n_mb:.2f} MB). "
        f"Se fusionan {_merge_ratio:.0f} oraciones por grupo → el DP solo ve {_n_sents_effective} unidades. "
        "Los cortes son poco precisos. Sube 'Máx. oraciones' o divide el texto."
    )
elif _will_merge:
    st.info(
        f"ℹ️ {_n_sents_raw:,} oraciones → fusión automática a {_n_sents_effective:,} grupos "
        f"(~{_merge_ratio:.1f} oraciones/grupo). "
        "Los cortes del DP caerán entre grupos, no entre oraciones individuales."
    )

# ── Botón principal: solo segmentación ────────────────────────────────────────
if st.button("Segmentar", type="primary"):
    with st.spinner("Calculando segmentaciones..."):
        try:
            dp_result = segment_dp(text, lmin=lmin, lmax=lmax, model=model, coherence_lambda=coherence_lambda, fixed_cost=fixed_cost, max_sentences=max_sentences)
            base_result = segment_baseline(text, lmin=lmin, lmax=lmax, model=model, max_sentences=max_sentences)
            sw_result = segment_sliding_window(text, lmax=lmax, overlap=overlap, model=model, max_sentences=max_sentences)
            overlap_result = segment_dp_overlap(text, lmin=lmin, lmax=lmax, model=model, coherence_lambda=coherence_lambda, overlap_mu=overlap_mu, max_overlap=max_overlap, max_sentences=max_sentences)
            tt_result = segment_texttiling(text, lmin=lmin, lmax=lmax, model=model, window_size=tt_window_size, smoothing_passes=tt_smoothing, max_sentences=max_sentences)
            report = compare(dp_result, base_result)
        except ValueError as e:
            st.error(f"Error: {e}")
            st.stop()

    # Guardar resultados en session_state para el botón LLM
    st.session_state["dp_result"] = dp_result
    st.session_state["base_result"] = base_result
    st.session_state["sw_result"] = sw_result
    st.session_state["overlap_result"] = overlap_result
    st.session_state["tt_result"] = tt_result
    st.session_state["report"] = report
    st.session_state["text"] = text
    st.session_state["llm_model"] = "gemma2:2b"
    st.session_state.pop("llm_evaluation", None)  # resetear evaluación LLM anterior

# ── Mostrar resultados si existen en session_state ────────────────────────────
if "dp_result" in st.session_state:
    dp_result = st.session_state["dp_result"]
    base_result = st.session_state["base_result"]
    sw_result = st.session_state["sw_result"]
    overlap_result = st.session_state["overlap_result"]
    tt_result = st.session_state["tt_result"]
    report = st.session_state["report"]

    # ── Métricas resumen ──────────────────────────────────────────────────────
    st.subheader("Resumen de métricas")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Segmentos DP", dp_result.num_segments)
    col2.metric("Segmentos Baseline", base_result.num_segments)
    col3.metric("Segmentos Sliding Window", sw_result.num_segments)
    col4.metric("Reducción de costo DP vs Baseline", f"{report.cost_reduction_pct}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo DP", f"{dp_result.total_cost:,.0f}")
    col2.metric("Costo Baseline", f"{base_result.total_cost:,.0f}")
    col3.metric("Costo Sliding Window", f"{sw_result.total_cost:,.0f}")
    col4.metric("Mejora coherencia", f"{report.coherence_improvement:+.4f}")

    # ── Visualización de segmentos ────────────────────────────────────────────
    st.subheader("Distribución de tokens por segmento")

    methods = [
        (dp_result.segments, "DP Óptimo", "#378ADD"),
        (base_result.segments, "Baseline", "#EF9F27"),
        (sw_result.segments, "Sliding Window", "#1D9E75"),
        (overlap_result.segments, "DP + Overlap", "#D85A30"),
        (tt_result.segments, "TextTiling", "#7B5EA7"),
    ]

    _bar_fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[title for _, title, _ in methods],
        horizontal_spacing=0.06,
    )

    for col_idx, (segs, title, color) in enumerate(methods, start=1):
        tokens = [s.token_count for s in segs]
        avg = float(np.mean(tokens))
        n = len(tokens)
        preview = [s.text[:90] + ("…" if len(s.text) > 90 else "") for s in segs]
        n_sents = [len(s.sentences) for s in segs]

        # Barras
        _bar_fig.add_trace(
            go.Bar(
                x=list(range(n)),
                y=tokens,
                marker_color=color,
                marker_opacity=0.85,
                marker_line_color="white",
                marker_line_width=0.8,
                customdata=list(zip(n_sents, preview)),
                hovertemplate=(
                    "<b>Segmento %{x}</b><br>"
                    "Tokens: <b>%{y}</b><br>"
                    "Oraciones: %{customdata[0]}<br>"
                    "<i style='color:#888'>%{customdata[1]}</i>"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=title,
            ),
            row=1, col=col_idx,
        )

        # Línea de promedio
        _bar_fig.add_trace(
            go.Scatter(
                x=[-0.5, n - 0.5],
                y=[avg, avg],
                mode="lines",
                line=dict(color="#d62728", dash="dash", width=1.8),
                showlegend=False,
                hovertemplate=f"promedio = {avg:.0f} tok<extra></extra>",
                name=f"promedio {title}",
            ),
            row=1, col=col_idx,
        )

        # Anotación del promedio
        _bar_fig.add_annotation(
            x=n - 1, y=avg,
            text=f"prom={avg:.0f}",
            showarrow=False, yanchor="bottom",
            font=dict(size=9, color="#d62728"),
            row=1, col=col_idx,
        )

    _bar_fig.update_layout(
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="#f8fafd",
        margin=dict(t=45, b=38, l=48, r=15),
        font=dict(family="Inter, sans-serif", size=13, color="#1f2937"),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#d1d5db",
            font=dict(size=14, family="Inter, sans-serif", color="#111827"),
        ),
    )
    _bar_fig.update_annotations(
        font=dict(color="#1f2937", size=13, family="Inter, sans-serif"),
    )
    _bar_fig.update_xaxes(
        title_text="Segmento #",
        tickfont=dict(size=12, color="#374151"),
        title_font=dict(color="#374151", size=13),
        gridcolor="#e5e7eb",
        linecolor="#d1d5db",
    )
    _bar_fig.update_yaxes(
        title_text="Tokens",
        tickfont=dict(size=12, color="#374151"),
        title_font=dict(color="#374151", size=13),
        gridcolor="#e5e7eb",
        zeroline=False,
        linecolor="#d1d5db",
    )

    st.plotly_chart(_bar_fig, use_container_width=True)

    # ── Visualización de cortes ───────────────────────────────────────────────
    st.subheader("Mapa de cortes sobre el texto")

    _PALETTE_BG = [
        "#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3", "#ede9fe",
        "#ffedd5", "#cffafe", "#ecfdf5", "#fef2f2", "#e0f2fe",
        "#f5f3ff", "#fdf4ff",
    ]
    _PALETTE_ACCENT = [
        "#3b82f6", "#22c55e", "#eab308", "#ec4899", "#8b5cf6",
        "#f97316", "#06b6d4", "#10b981", "#ef4444", "#0ea5e9",
        "#7c3aed", "#a21caf",
    ]

    def render_colored_text(segs):
        import html as _html
        legend = '<div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px">'
        for i, seg in enumerate(segs):
            bg = _PALETTE_BG[i % len(_PALETTE_BG)]
            ac = _PALETTE_ACCENT[i % len(_PALETTE_ACCENT)]
            legend += (
                f'<span style="background:{bg}; border:1.5px solid {ac}; color:{ac}; '
                f'font-size:10.5px; font-weight:700; padding:2px 9px; border-radius:99px">'
                f'S{seg.index} · {seg.token_count} tok</span>'
            )
        legend += "</div>"

        body = (
            '<div style="font-size:0.875rem; line-height:1.85; padding:14px 16px; '
            'background:#fafafa; border:1px solid #e5e7eb; border-radius:8px; '
            'font-family:Inter,sans-serif; color:#1f2937; word-break:break-word">'
        )
        for i, seg in enumerate(segs):
            bg = _PALETTE_BG[i % len(_PALETTE_BG)]
            ac = _PALETTE_ACCENT[i % len(_PALETTE_ACCENT)]
            raw = _html.escape(seg.text)
            body += (
                f'<span style="background:{bg}; border-bottom:2px solid {ac}; '
                f'padding:1px 2px; border-radius:3px" '
                f'title="S{seg.index} · {seg.token_count} tokens">'
                f'<span style="display:inline-block; background:{ac}; color:#fff; '
                f'font-size:9px; font-weight:800; padding:1px 5px; border-radius:3px; '
                f'margin-right:3px; vertical-align:middle; line-height:14px">S{seg.index}</span>'
                f'{raw}</span>'
            )
        body += "</div>"
        return legend + body

    def render_segments(segs, color):
        import html as _html
        preview_len = 90
        total_tokens = sum(s.token_count for s in segs)
        html = '<div style="display:flex; flex-wrap:wrap; gap:8px; padding:6px 0">'
        cumulative = 0
        for seg in segs:
            raw = seg.text[:preview_len] + ("…" if len(seg.text) > preview_len else "")
            preview = _html.escape(raw)
            token_start = cumulative
            token_end = cumulative + seg.token_count - 1
            left_pct = (token_start / total_tokens * 100) if total_tokens else 0
            width_pct = (seg.token_count / total_tokens * 100) if total_tokens else 100
            cumulative += seg.token_count
            html += f"""
            <div style="
                display:inline-block;
                background:#ffffff;
                border:1px solid #e5e7eb;
                border-left:4px solid {color};
                border-radius:8px;
                padding:9px 13px;
                min-width:160px;
                max-width:280px;
                box-shadow:0 1px 4px rgba(0,0,0,0.07);
                vertical-align:top;
            ">
                <div style="display:flex; align-items:center; gap:7px; margin-bottom:5px">
                    <span style="font-weight:700; font-size:13px; color:#0f2540">S{seg.index}</span>
                    <span style="
                        background:{color}1a;
                        color:{color};
                        border:1px solid {color}55;
                        font-size:10.5px;
                        font-weight:600;
                        padding:1px 7px;
                        border-radius:99px;
                    ">{seg.token_count} tok</span>
                </div>
                <span style="font-size:11.5px; color:#1f2937; line-height:1.45; display:block">{preview}</span>
                <div style="margin-top:7px">
                    <div style="font-size:9.5px; color:#6b7280; margin-bottom:3px; font-family:monospace">
                        tok {token_start} → {token_end}
                    </div>
                    <div style="background:#e5e7eb; border-radius:99px; height:5px; position:relative; overflow:hidden">
                        <div style="
                            position:absolute;
                            left:{left_pct:.2f}%;
                            width:max({width_pct:.2f}%, 4px);
                            height:100%;
                            background:{color};
                            border-radius:99px;
                            opacity:0.9;
                        "></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:8.5px; color:#9ca3af; margin-top:2px">
                        <span>0</span>
                        <span>{total_tokens} tok</span>
                    </div>
                </div>
            </div>
            """
        html += "</div>"
        return html

    def _tok_chip(tok_str, abs_idx, tid, bg, fg, extra_style=""):
        import html as _html
        escaped = _html.escape(tok_str).replace(" ", "&nbsp;")
        return (
            f'<span style="background:{bg}; color:{fg}; border-radius:3px; '
            f'padding:2px 4px; margin:0 1px; font-size:0.81rem; font-family:monospace; '
            f'display:inline-block; cursor:default; white-space:pre;{extra_style}" '
            f'title="tok #{abs_idx} · id {tid}">{escaped}</span>'
        )

    def render_tokenized_sentences(seg, seg_token_start: int, model_name: str) -> str:
        from src.segmentation.tokenizer import get_encoder
        enc = get_encoder(model_name)
        TOKEN_BG = [
            "#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3", "#ede9fe",
            "#ffedd5", "#cffafe", "#ecfdf5", "#fef2f2", "#e0f2fe",
            "#f5f3ff", "#fdf4ff",
        ]
        TOKEN_FG = [
            "#1d4ed8", "#15803d", "#854d0e", "#9d174d", "#5b21b6",
            "#9a3412", "#0e7490", "#065f46", "#991b1b", "#0369a1",
            "#6d28d9", "#7e22ce",
        ]

        def decode_token(tid):
            raw = enc.decode_single_token_bytes(tid)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace")

        out = ""

        # ── Sección de overlap (tokens repetidos del segmento anterior) ──────
        overlap_sents = getattr(seg, "overlap_sentences", [])
        overlap_toks  = getattr(seg, "overlap_tokens", 0)
        if overlap_sents and overlap_toks > 0:
            ov_global = seg_token_start - overlap_toks
            out += (
                '<div style="background:#f8fafc; border:1px dashed #cbd5e1; '
                'border-radius:6px; padding:8px 10px; margin-bottom:10px">'
                '<div style="font-size:10px; color:#64748b; font-weight:700; '
                'margin-bottom:6px; letter-spacing:0.3px">↩ OVERLAP — tokens compartidos con el segmento anterior</div>'
            )
            for ov_sent in overlap_sents:
                ids = enc.encode(ov_sent)
                out += '<div style="line-height:2.4; word-break:break-word; margin-bottom:4px">'
                for i, tid in enumerate(ids):
                    abs_idx = ov_global + i
                    out += _tok_chip(
                        decode_token(tid), abs_idx, tid,
                        bg="#f1f5f9", fg="#64748b",
                        extra_style="border:1px dashed #94a3b8; opacity:0.85;",
                    )
                out += "</div>"
                ov_global += len(ids)
            out += "</div>"

        # ── Oraciones propias del segmento ────────────────────────────────────
        global_tok = seg_token_start
        for j, sent in enumerate(seg.sentences):
            ids = enc.encode(sent)
            n_tok = len(ids)
            out += (
                f'<div style="margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f1f5f9">'
                f'<div style="font-size:10px; color:#94a3b8; font-weight:600; margin-bottom:5px">'
                f'Oración {j} &nbsp;·&nbsp; {n_tok} tokens &nbsp;·&nbsp; tok {global_tok} → {global_tok + n_tok - 1}'
                f'</div><div style="line-height:2.4; word-break:break-word">'
            )
            for i, tid in enumerate(ids):
                abs_idx = global_tok + i
                ci = abs_idx % len(TOKEN_BG)
                out += _tok_chip(decode_token(tid), abs_idx, tid, TOKEN_BG[ci], TOKEN_FG[ci])
            out += "</div></div>"
            global_tok += n_tok
        return out

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["DP Óptimo", "Baseline", "Sliding Window", "DP + Overlap", "TextTiling"])

    with tab1:
        st.html(render_segments(dp_result.segments, "#378ADD"))
        _cum = 0
        for seg in dp_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — {len(seg.sentences)} oraciones"):
                st.html(render_tokenized_sentences(seg, _cum, model))
            _cum += seg.token_count

    with tab2:
        st.html(render_segments(base_result.segments, "#EF9F27"))
        _cum = 0
        for seg in base_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — {len(seg.sentences)} oraciones"):
                st.html(render_tokenized_sentences(seg, _cum, model))
            _cum += seg.token_count

    with tab3:
        st.html(render_segments(sw_result.segments, "#1D9E75"))
        _cum = 0
        for seg in sw_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — solapamiento: {seg.overlap_tokens} tok"):
                st.html(render_tokenized_sentences(seg, _cum, model))
            _cum += seg.token_count

    with tab4:
        st.html(render_segments(overlap_result.segments, "#D85A30"))
        _cum = 0
        for seg in overlap_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — solapamiento: {seg.overlap_tokens} tok"):
                st.html(render_tokenized_sentences(seg, _cum, model))
            _cum += seg.token_count

    with tab5:
        st.html(render_segments(tt_result.segments, "#7B5EA7"))
        _cum = 0
        for seg in tt_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — {len(seg.sentences)} oraciones"):
                st.html(render_tokenized_sentences(seg, _cum, model))
            _cum += seg.token_count

    # ── Vista continua coloreada ──────────────────────────────────────────────
    st.subheader("Vista continua del texto por segmentos")
    st.caption("El texto completo pintado por segmento. Cada color corresponde a un segmento; la etiqueta Sₙ indica el índice.")

    ct1, ct2, ct3, ct4, ct5 = st.tabs(["DP Óptimo", "Baseline", "Sliding Window", "DP + Overlap", "TextTiling"])
    with ct1:
        st.html(render_colored_text(dp_result.segments))
    with ct2:
        st.html(render_colored_text(base_result.segments))
    with ct3:
        st.html(render_colored_text(sw_result.segments))
    with ct4:
        st.html(render_colored_text(overlap_result.segments))
    with ct5:
        st.html(render_colored_text(tt_result.segments))

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    st.subheader("Tabla comparativa")
    dp_m = compute_metrics(dp_result)
    base_m = compute_metrics(base_result)
    sw_m = compute_metrics(sw_result)

    # ── Curva en U — costo por número de segmentos ────────────────────────────
    st.subheader("Curva en U — costo total vs número de segmentos (DP 2D)")

    with st.spinner("Calculando DP 2D..."):
        try:
            # Se calcula primero con el costo fijo REAL (el mismo de la comparación de arriba),
            # para que el k* de la curva coincida con los segmentos del DP.
            dp2d_result = segment_dp_2d(
                text, lmin=lmin, lmax=lmax, model=model,
                coherence_lambda=coherence_lambda,
                fixed_cost=fixed_cost,
                max_sentences=max_sentences,
            )
            _ks_sorted = sorted(int(k) for k in dp2d_result.cost_by_k)
            # Si el óptimo cae en el extremo derecho del rango (típico en textos cortos),
            # la curva parece una "L" (solo bajada). Solo en ese caso se sube a un costo
            # fijo didáctico para que la forma en U sea evidente.
            _ucurve_didactic = (
                len(_ks_sorted) >= 2 and dp2d_result.num_segments >= _ks_sorted[-1] - 1
            )
            if _ucurve_didactic:
                dp2d_result = segment_dp_2d(
                    text, lmin=lmin, lmax=lmax, model=model,
                    coherence_lambda=coherence_lambda,
                    fixed_cost=_UCURVE_FIXED_COST,
                    max_sentences=max_sentences,
                )
                st.caption(
                    "📈 Curva **ilustrativa** del principio del óptimo k\\*: pocos segmentos "
                    "(atención cuadrática cara) y demasiados (mucho costo fijo) son ambos costosos. "
                    f"Como el texto es corto, para que la **U** se vea se usa un costo fijo didáctico "
                    f"de **{int(_UCURVE_FIXED_COST)}**; por eso el k\\* de la curva es menor que los "
                    f"segmentos del DP de arriba (que usan Cf={int(fixed_cost)})."
                )
            else:
                st.caption(
                    f"📈 Costo total para cada número de segmentos k, con el costo fijo actual "
                    f"(**{int(fixed_cost)}**). El mínimo **k\\*** coincide con los segmentos del DP de arriba."
                )

            k_values = list(dp2d_result.cost_by_k.keys())
            cost_values = list(dp2d_result.cost_by_k.values())
            best_k = dp2d_result.num_segments
            best_cost = dp2d_result.total_cost

            _ucurve_fig = go.Figure()

            # Área rellena bajo la curva
            _ucurve_fig.add_trace(go.Scatter(
                x=k_values, y=cost_values,
                mode="none",
                fill="tozeroy",
                fillcolor="rgba(55,138,221,0.08)",
                showlegend=False,
                hoverinfo="skip",
            ))

            # Línea principal con marcadores
            _ucurve_fig.add_trace(go.Scatter(
                x=k_values, y=cost_values,
                mode="lines+markers",
                name="Costo total",
                line=dict(color="#378ADD", width=2.5),
                marker=dict(size=8, color="white",
                            line=dict(color="#378ADD", width=2.5)),
                hovertemplate="<b>k = %{x}</b><br>Costo: %{y:,.0f} tok²<extra></extra>",
            ))

            # Punto óptimo destacado
            _ucurve_fig.add_trace(go.Scatter(
                x=[best_k], y=[best_cost],
                mode="markers",
                name=f"k* = {best_k}",
                marker=dict(size=14, color="#d62728",
                            line=dict(color="white", width=2.5)),
                hovertemplate=f"<b>k óptimo = {best_k}</b><br>Costo: {best_cost:,.0f} tok²<extra></extra>",
            ))

            # Línea vertical en k óptimo
            _ucurve_fig.add_vline(
                x=best_k,
                line_dash="dash", line_color="#d62728", line_width=1.6,
                annotation_text=f"  k* = {best_k}",
                annotation_position="top right",
                annotation_font=dict(color="#d62728", size=14),
            )

            _ucurve_fig.update_layout(
                title=dict(text="Costo total vs número de segmentos k",
                           font=dict(size=17, family="Inter, sans-serif",
                                     color="#0f2540")),
                xaxis=dict(
                    title="k (número de segmentos)",
                    title_font=dict(color="#374151", size=14),
                    tickvals=k_values,
                    tickfont=dict(color="#374151", size=13),
                    gridcolor="#e5e7eb",
                    linecolor="#d1d5db",
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Costo total (tokens²)",
                    title_font=dict(color="#374151", size=14),
                    tickfont=dict(color="#374151", size=13),
                    gridcolor="#e5e7eb",
                    linecolor="#d1d5db",
                    zeroline=False,
                ),
                height=440,
                paper_bgcolor="white",
                plot_bgcolor="#f8fafd",
                margin=dict(t=62, b=52, l=82, r=30),
                font=dict(family="Inter, sans-serif", size=13, color="#1f2937"),
                legend=dict(x=0.75, y=0.95, bgcolor="rgba(255,255,255,0.92)",
                            bordercolor="#bfd3ef", borderwidth=1,
                            font=dict(color="#1f2937", size=13)),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="#d1d5db",
                    font=dict(size=14, color="#111827"),
                ),
            )

            st.plotly_chart(_ucurve_fig, use_container_width=True)
            _cf_label = (
                f"costo fijo didáctico = {int(_UCURVE_FIXED_COST)}"
                if _ucurve_didactic
                else f"costo fijo = {int(fixed_cost)} (igual que la comparación)"
            )
            st.caption(
                f"k\\* óptimo de la curva = {best_k} segmentos · costo mínimo: {best_cost:,.0f} tok² "
                f"({_cf_label})"
            )

            with st.expander("Ver tabla completa de costos por k"):
                st.table({
                    "k (segmentos)": k_values,
                    "Costo total (tokens²)": [f"{c:,.0f}" for c in cost_values],
                    "Óptimo": ["✓" if k == best_k else "" for k in k_values],
                })

        except ValueError as e:
            st.warning(f"DP 2D: {e}")

    # ── Calibración analítica ─────────────────────────────────────────────────
    st.subheader("Calibración de parámetros")

    with st.spinner("Calculando calibración..."):
        cal_lambda = suggest_lambda(st.session_state["text"])
        cal_k_star = estimate_optimal_k(
            st.session_state["text"],
            model=model,
            fixed_cost=fixed_cost,
            coherence_lambda=coherence_lambda,
        )
        cal_result = validate_calibration(
            st.session_state["text"],
            lmin=lmin, lmax=lmax,
            coherence_lambda=coherence_lambda,
            fixed_cost=fixed_cost,
            model=model,
            max_sentences=max_sentences,
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("λ sugerido", cal_lambda,
                help="Valor óptimo de λ según variabilidad de coherencia del texto")
    col2.metric("k* analítico", cal_k_star,
                help="Número óptimo de segmentos estimado por fórmula: k* = n/√Cf")
    col3.metric("k* DP 2D", cal_result["k_dp"],
                help="Número óptimo encontrado por el algoritmo DP 2D")
    col4.metric(
        "Desviación k*",
        f"{cal_result['deviation_pct']:.1%}",
        delta="válida ✓" if cal_result["valid"] else "revisar ✗",
        delta_color="normal" if cal_result["valid"] else "inverse",
        help="Desviación entre k* analítico y k* DP. ≤20% es válida."
    )

    if not cal_result["valid"]:
        st.warning(
            f"Calibración fuera de rango (desviación {cal_result['deviation_pct']:.1%}). "
            "Ajusta `fixed_cost` o `lmin/lmax` para acercar k* analítico al k* DP."
        )

    tt_m = compute_metrics(tt_result)
    overlap_m = compute_metrics(overlap_result)

    # Métricas de frontera: Pk y WindowDiff usando DP como referencia óptima
    try:
        _dp_bounds  = boundaries_from_result(dp_result)
        _pk_base    = pk(_dp_bounds, boundaries_from_result(base_result))
        _pk_sw      = pk(_dp_bounds, boundaries_from_result(sw_result))
        _pk_ov      = pk(_dp_bounds, boundaries_from_result(overlap_result))
        _pk_tt      = pk(_dp_bounds, boundaries_from_result(tt_result))
        _wd_base    = window_diff(_dp_bounds, boundaries_from_result(base_result))
        _wd_sw      = window_diff(_dp_bounds, boundaries_from_result(sw_result))
        _wd_ov      = window_diff(_dp_bounds, boundaries_from_result(overlap_result))
        _wd_tt      = window_diff(_dp_bounds, boundaries_from_result(tt_result))
        _pk_col = ["0.000", f"{_pk_base:.3f}", f"{_pk_sw:.3f}", f"{_pk_ov:.3f}", f"{_pk_tt:.3f}"]
        _wd_col = ["0.000", f"{_wd_base:.3f}", f"{_wd_sw:.3f}", f"{_wd_ov:.3f}", f"{_wd_tt:.3f}"]
    except Exception:
        _pk_col = ["—"] * 5
        _wd_col = ["—"] * 5

    _ov_total_sw  = getattr(sw_result,      "total_overlap_tokens", 0)
    _ov_total_ov  = getattr(overlap_result, "total_overlap_tokens", 0)

    _table_data = {
        "Método":          ["DP Óptimo", "Baseline", "Sliding Window", "DP + Overlap", "TextTiling"],
        "Segmentos":       [str(dp_result.num_segments), str(base_result.num_segments),
                            str(sw_result.num_segments), str(overlap_result.num_segments),
                            str(tt_result.num_segments)],
        "Costo total":     [f"{dp_result.total_cost:,.0f}", f"{base_result.total_cost:,.0f}",
                            f"{sw_result.total_cost:,.0f}", f"{overlap_result.total_cost:,.0f}",
                            f"{tt_result.total_cost:,.0f}"],
        "Tokens total":    [str(dp_m.total_tokens), str(base_m.total_tokens),
                            str(sw_m.total_tokens), str(overlap_m.total_tokens),
                            str(tt_m.total_tokens)],
        "Tokens prom.":    [f"{dp_m.avg_tokens_per_segment:.1f}", f"{base_m.avg_tokens_per_segment:.1f}",
                            f"{sw_m.avg_tokens_per_segment:.1f}", f"{overlap_m.avg_tokens_per_segment:.1f}",
                            f"{tt_m.avg_tokens_per_segment:.1f}"],
        "Tokens mín.":     [str(dp_m.min_tokens), str(base_m.min_tokens),
                            str(sw_m.min_tokens), str(overlap_m.min_tokens),
                            str(tt_m.min_tokens)],
        "Tokens máx.":     [str(dp_m.max_tokens), str(base_m.max_tokens),
                            str(sw_m.max_tokens), str(overlap_m.max_tokens),
                            str(tt_m.max_tokens)],
        "Desv. tokens":    [f"{dp_m.std_tokens_per_segment:.1f}", f"{base_m.std_tokens_per_segment:.1f}",
                            f"{sw_m.std_tokens_per_segment:.1f}", f"{overlap_m.std_tokens_per_segment:.1f}",
                            f"{tt_m.std_tokens_per_segment:.1f}"],
        "Solapamiento total": ["—", "—", str(_ov_total_sw), str(_ov_total_ov), "—"],
        "Coherencia":      [f"{dp_m.avg_coherence:.4f}", f"{base_m.avg_coherence:.4f}",
                            f"{sw_m.avg_coherence:.4f}", f"{overlap_m.avg_coherence:.4f}",
                            f"{tt_m.avg_coherence:.4f}"],
        "Pk↓ (vs DP)":     _pk_col,
        "WD↓ (vs DP)":     _wd_col,
    }
    st.dataframe(_table_data, use_container_width=True, hide_index=True)
    st.caption(
        "**Pk** (Beeferman 1999) y **WD** (WindowDiff, Pevzner & Hearst 2002): "
        "desacuerdo de fronteras respecto al DP Óptimo · 0 = acuerdo perfecto · valores menores = mejor"
    )

    # ── Descarga de resultados ────────────────────────────────────────────────
    st.divider()
    st.subheader("Descargar resultados")
    col_dl1, col_dl2 = st.columns(2)

    def _fmt_segments_txt(result, label):
        lines = [f"=== {label} — {result.num_segments} segmentos ===\n"]
        for seg in result.segments:
            lines.append(f"--- Segmento {seg.index} ({seg.token_count} tokens) ---")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    with col_dl1:
        txt_output = "\n\n".join([
            _fmt_segments_txt(dp_result, "DP Óptimo"),
            _fmt_segments_txt(base_result, "Baseline"),
            _fmt_segments_txt(tt_result, "TextTiling"),
        ])
        st.download_button(
            "⬇ Segmentaciones (.txt)",
            data=txt_output,
            file_name="segmentaciones.txt",
            mime="text/plain",
            help="Descarga los segmentos de DP, Baseline y TextTiling en texto plano.",
        )

    with col_dl2:
        def _metrics_dict(m):
            return {
                "num_segments": m.num_segments,
                "total_cost": m.total_cost,
                "total_tokens": m.total_tokens,
                "avg_tokens": m.avg_tokens_per_segment,
                "std_tokens": m.std_tokens_per_segment,
                "min_tokens": m.min_tokens,
                "max_tokens": m.max_tokens,
                "avg_coherence": m.avg_coherence,
            }
        json_output = json.dumps({
            "params": {
                "lmin": lmin, "lmax": lmax,
                "coherence_lambda": coherence_lambda,
                "fixed_cost": fixed_cost, "model": model,
                "overlap": overlap, "overlap_mu": overlap_mu,
                "max_overlap": max_overlap,
                "tt_window_size": tt_window_size,
                "tt_smoothing_passes": tt_smoothing,
            },
            "dp": {
                "metrics": _metrics_dict(dp_m),
                "segments": [{"index": s.index, "text": s.text,
                               "token_count": s.token_count, "sentences": s.sentences}
                              for s in dp_result.segments],
            },
            "baseline": {
                "metrics": _metrics_dict(base_m),
                "segments": [{"index": s.index, "text": s.text,
                               "token_count": s.token_count, "sentences": s.sentences}
                              for s in base_result.segments],
            },
            "sliding_window": {
                "total_overlap_tokens": getattr(sw_result, "total_overlap_tokens", 0),
                "metrics": _metrics_dict(sw_m),
                "segments": [{"index": s.index, "text": s.text, "token_count": s.token_count,
                               "overlap_tokens": s.overlap_tokens,
                               "overlap_sentences": getattr(s, "overlap_sentences", [])}
                              for s in sw_result.segments],
            },
            "dp_overlap": {
                "total_overlap_tokens": getattr(overlap_result, "total_overlap_tokens", 0),
                "metrics": _metrics_dict(overlap_m),
                "segments": [{"index": s.index, "text": s.text, "token_count": s.token_count,
                               "overlap_tokens": s.overlap_tokens,
                               "overlap_sentences": s.overlap_sentences}
                              for s in overlap_result.segments],
            },
            "texttiling": {
                "metrics": _metrics_dict(tt_m),
                "segments": [{"index": s.index, "text": s.text,
                               "token_count": s.token_count, "sentences": s.sentences}
                              for s in tt_result.segments],
            },
        }, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇ Resultados completos (.json)",
            data=json_output,
            file_name="segmentacion_resultados.json",
            mime="application/json",
            help="Descarga todos los resultados con metadatos en JSON.",
        )

    # ── Evaluación LLM — botón separado ──────────────────────────────────────
    st.divider()
    st.subheader("Evaluación con LLM real")
    st.caption("Ejecuta inferencia local con Ollama. Requiere que `gemma2:2b` esté descargado.")

    if st.button("Evaluar con LLM real (lento)", type="secondary"):
        with st.spinner("Ejecutando inferencia con gemma2:2b..."):
            try:
                dp_segments = [s.text for s in dp_result.segments]
                base_segments = [s.text for s in base_result.segments]
                sw_segments = [s.text for s in sw_result.segments]

                llm_dp = evaluate_segmentation(dp_segments, method="dp", model="gemma2:2b")
                llm_base = evaluate_segmentation(base_segments, method="baseline", model="gemma2:2b")
                llm_sw = evaluate_segmentation(sw_segments, method="sliding_window", model="gemma2:2b")

                st.session_state["llm_evaluation"] = {
                    "dp": llm_dp,
                    "baseline": llm_base,
                    "sliding_window": llm_sw,
                }
            except Exception as e:
                st.error(f"Error al conectar con Ollama: {e}")

    if "llm_evaluation" in st.session_state:
        llm = st.session_state["llm_evaluation"]

        st.subheader("Resultados de evaluación LLM")

        # evaluate_segmentation devuelve un dataclass EvaluationReport (no un dict),
        # por eso se accede con getattr y no con .get().
        def _llm_metric(report, attr: str, fmt: str = ".2f"):
            val = getattr(report, attr, None)
            if val is None:
                return "—"
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return str(val)

        def _llm_avg_prompt(report):
            n = getattr(report, "num_segments", 0) or 0
            total = getattr(report, "total_prompt_tokens", 0) or 0
            return f"{total / n:.0f}" if n else "—"

        def _llm_col(title, report):
            st.markdown(f"**{title}**")
            st.metric("Latencia prom. (ms)", _llm_metric(report, "avg_latency_ms", ".0f"))
            st.metric("Tokens prompt prom.", _llm_avg_prompt(report))
            st.metric("Coherencia LLM prom.", _llm_metric(report, "avg_response_coherence", ".3f"))

        col1, col2, col3 = st.columns(3)
        with col1:
            _llm_col("DP Óptimo", llm["dp"])
        with col2:
            _llm_col("Baseline", llm["baseline"])
        with col3:
            _llm_col("Sliding Window", llm["sliding_window"])