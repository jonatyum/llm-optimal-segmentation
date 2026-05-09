import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation.dp import segment_dp, segment_dp_2d
from src.segmentation.baseline import segment_baseline
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.texttiling import segment_texttiling
from src.segmentation.metrics import compute_metrics, compare
from src.segmentation.evaluator import evaluate_segmentation
from src.segmentation.calibration import suggest_lambda, estimate_optimal_k, validate_calibration

st.set_page_config(
    page_title="LLM Optimal Segmentation",
    page_icon="✂️",
    layout="wide"
)

st.title("LLM Optimal Segmentation")
st.caption("Comparación visual de métodos de segmentación — Tesis UMSA 2025")

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
        50, 400, 100, step=10,
        help="Límite superior de tokens por segmento. Representa la ventana de contexto disponible del LLM."
    )
    overlap = st.slider(
        "Overlap — sliding window",
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
        0.0, 2000.0, 1000.0, step=50.0,
        help="Overhead fijo por cada llamada al LLM. Valores altos favorecen menos segmentos más grandes. Produce la curva en U en la DP 2D."
    )
    model = st.selectbox(
        "Modelo de tokenización",
        ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        help="Modelo cuyo tokenizador se usa para contar tokens. Afecta la distribución de tokens por segmento."
    )

    st.divider()
    st.caption("**Guía rápida**")
    st.caption("↑ lmax → más tokens por segmento")
    st.caption("↑ λ → más coherencia semántica")
    st.caption("↑ fixed_cost → curva en U más pronunciada")

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Texto de entrada")
default_text = (
    "Large language models have billions of parameters and require significant "
    "computational resources for inference. Techniques like quantization and "
    "distillation help reduce inference costs substantially. Prompt engineering "
    "has emerged as a key skill for working with these models effectively. "
    "Retrieval-augmented generation combines LLMs with external knowledge sources "
    "to improve factual accuracy. Fine-tuning on domain-specific data can "
    "substantially improve performance on specialized tasks. Evaluation remains "
    "a challenge, as automated metrics often fail to capture true quality. "
    "The transformer architecture introduced self-attention mechanisms that allow "
    "models to weigh the importance of different tokens dynamically. Unlike "
    "recurrent networks, transformers process all tokens in parallel, which "
    "dramatically reduces training time on modern hardware accelerators. "
    "BERT leveraged bidirectional pre-training to achieve state-of-the-art results. "
    "GPT models use a unidirectional autoregressive approach instead, making them "
    "particularly well-suited for open-ended text generation tasks."
)
text = st.text_area("Ingresa el texto a segmentar", value=default_text, height=180)

# ── Botón principal: solo segmentación ────────────────────────────────────────
if st.button("Segmentar", type="primary"):
    with st.spinner("Calculando segmentaciones..."):
        try:
            dp_result = segment_dp(text, lmin=lmin, lmax=lmax, model=model, coherence_lambda=coherence_lambda)
            base_result = segment_baseline(text, lmin=lmin, lmax=lmax, model=model)
            sw_result = segment_sliding_window(text, lmax=lmax, overlap=overlap, model=model)
            overlap_result = segment_dp_overlap(text, lmin=lmin, lmax=lmax, model=model, coherence_lambda=coherence_lambda, overlap_mu=overlap_mu)
            tt_result = segment_texttiling(text, lmin=lmin, lmax=lmax, model=model)
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
    fig, axes = plt.subplots(1, 5, figsize=(20, 3))

    methods = [
        (dp_result.segments, "DP Optimal", "#378ADD"),
        (base_result.segments, "Baseline", "#EF9F27"),
        (sw_result.segments, "Sliding Window", "#1D9E75"),
        (overlap_result.segments, "DP + Overlap", "#D85A30"),
        (tt_result.segments, "TextTiling", "#7B5EA7"),
    ]

    for ax, (segs, title, color) in zip(axes, methods):
        tokens = [s.token_count for s in segs]
        ax.bar(range(len(tokens)), tokens, color=color)
        ax.axhline(y=np.mean(tokens), color="red", linestyle="--", linewidth=1, label=f"avg={np.mean(tokens):.0f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Segmento")
        ax.set_ylabel("Tokens")
        ax.set_xticks(range(len(tokens)))
        ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # ── Visualización de cortes ───────────────────────────────────────────────
    st.subheader("Mapa de cortes sobre el texto")

    def render_segments(segs, color):
        html = ""
        colors = [
            "#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3",
            "#ede9fe", "#ffedd5", "#f1f5f9", "#cffafe",
        ]
        for i, seg in enumerate(segs):
            bg = colors[i % len(colors)]
            html += f"""
            <div style="display:inline-block; background:{bg}; border:1px solid #ccc;
                        border-radius:6px; padding:6px 10px; margin:4px;
                        font-size:12px; max-width:100%">
                <span style="font-weight:600; color:#333">S{seg.index}</span>
                <span style="color:#666; margin-left:6px">{seg.token_count} tok</span><br>
                <span style="color:#444">{seg.text[:80]}{'...' if len(seg.text) > 80 else ''}</span>
            </div>
            """
        return html

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["DP Optimal", "Baseline", "Sliding Window", "DP + Overlap", "TextTiling"])

    with tab1:
        st.markdown(render_segments(dp_result.segments, "#378ADD"), unsafe_allow_html=True)
        for seg in dp_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — {len(seg.sentences)} oraciones"):
                for j, sent in enumerate(seg.sentences):
                    st.markdown(f"**S{j}:** {sent}")

    with tab2:
        st.markdown(render_segments(base_result.segments, "#EF9F27"), unsafe_allow_html=True)
        for seg in base_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — {len(seg.sentences)} oraciones"):
                for j, sent in enumerate(seg.sentences):
                    st.markdown(f"**S{j}:** {sent}")

    with tab3:
        st.markdown(render_segments(sw_result.segments, "#1D9E75"), unsafe_allow_html=True)
        for seg in sw_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — overlap: {seg.overlap_tokens} tok"):
                for j, sent in enumerate(seg.sentences):
                    st.markdown(f"**S{j}:** {sent}")

    with tab4:
        st.markdown(render_segments(overlap_result.segments, "#D85A30"), unsafe_allow_html=True)
        for seg in overlap_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — overlap: {seg.overlap_tokens} tok"):
                for j, sent in enumerate(seg.sentences):
                    st.markdown(f"**S{j}:** {sent}")

    with tab5:
        st.markdown(render_segments(tt_result.segments, "#7B5EA7"), unsafe_allow_html=True)
        for seg in tt_result.segments:
            with st.expander(f"Segmento {seg.index} — {seg.token_count} tokens — {len(seg.sentences)} oraciones"):
                for j, sent in enumerate(seg.sentences):
                    st.markdown(f"**S{j}:** {sent}")

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    st.subheader("Tabla comparativa")
    dp_m = compute_metrics(dp_result)
    base_m = compute_metrics(base_result)
    sw_m = compute_metrics(sw_result)

    # ── Curva en U — costo por número de segmentos ────────────────────────────
    st.subheader("Curva de costo por número de segmentos (DP 2D)")

    with st.spinner("Calculando DP 2D..."):
        try:
            dp2d_result = segment_dp_2d(
                text, lmin=lmin, lmax=lmax, model=model,
                coherence_lambda=coherence_lambda,
                fixed_cost=fixed_cost
            )

            k_values = list(dp2d_result.cost_by_k.keys())
            cost_values = list(dp2d_result.cost_by_k.values())
            best_k = dp2d_result.num_segments

            fig2, ax = plt.subplots(figsize=(10, 4))
            ax.plot(k_values, cost_values, marker="o", color="#378ADD", linewidth=2)
            ax.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5, label=f"k óptimo = {best_k}")
            ax.scatter([best_k], [dp2d_result.total_cost], color="red", zorder=5, s=100)
            ax.set_title("Costo total vs número de segmentos k")
            ax.set_xlabel("k (número de segmentos)")
            ax.set_ylabel("Costo total (tokens²)")
            ax.set_xticks(k_values)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig2)

            st.caption(f"k óptimo = {best_k} segmentos con costo {dp2d_result.total_cost:,.0f} tokens²")

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
        cal_k_star = estimate_optimal_k(st.session_state["text"], fixed_cost=fixed_cost)
        cal_result = validate_calibration(
            st.session_state["text"],
            lmin=lmin, lmax=lmax,
            coherence_lambda=coherence_lambda,
            fixed_cost=fixed_cost,
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
    st.table({
        "Método": ["DP Optimal", "Baseline", "Sliding Window", "DP + Overlap", "TextTiling"],
        "Segmentos": [dp_result.num_segments, base_result.num_segments, sw_result.num_segments, overlap_result.num_segments, tt_result.num_segments],
        "Costo total": [f"{dp_result.total_cost:,.0f}", f"{base_result.total_cost:,.0f}", f"{sw_result.total_cost:,.0f}", f"{overlap_result.total_cost:,.0f}", f"{tt_result.total_cost:,.0f}"],
        "Avg tokens": [dp_m.avg_tokens_per_segment, base_m.avg_tokens_per_segment, sw_m.avg_tokens_per_segment, "—", tt_m.avg_tokens_per_segment],
        "Std tokens": [dp_m.std_tokens_per_segment, base_m.std_tokens_per_segment, sw_m.std_tokens_per_segment, "—", tt_m.std_tokens_per_segment],
        "Coherencia": [dp_m.avg_coherence, base_m.avg_coherence, sw_m.avg_coherence, "—", tt_m.avg_coherence],
    })

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

        def _llm_metric(eval_dict: dict, key: str, fmt: str = ".2f"):
            val = eval_dict.get(key, None)
            if val is None:
                return "—"
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return str(val)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**DP Optimal**")
            st.metric("Latencia avg (ms)", _llm_metric(llm["dp"], "avg_latency_ms"))
            st.metric("Tokens prompt avg", _llm_metric(llm["dp"], "avg_prompt_tokens", ".0f"))
            st.metric("Coherencia LLM avg", _llm_metric(llm["dp"], "avg_coherence"))

        with col2:
            st.markdown("**Baseline**")
            st.metric("Latencia avg (ms)", _llm_metric(llm["baseline"], "avg_latency_ms"))
            st.metric("Tokens prompt avg", _llm_metric(llm["baseline"], "avg_prompt_tokens", ".0f"))
            st.metric("Coherencia LLM avg", _llm_metric(llm["baseline"], "avg_coherence"))

        with col3:
            st.markdown("**Sliding Window**")
            st.metric("Latencia avg (ms)", _llm_metric(llm["sliding_window"], "avg_latency_ms"))
            st.metric("Tokens prompt avg", _llm_metric(llm["sliding_window"], "avg_prompt_tokens", ".0f"))
            st.metric("Coherencia LLM avg", _llm_metric(llm["sliding_window"], "avg_coherence"))