import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation.dp import segment_dp
from src.segmentation.baseline import segment_baseline
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.metrics import compute_metrics, compare

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
    lmin = st.slider("Lmin (tokens mínimos)", 5, 100, 20, step=5)
    lmax = st.slider("Lmax (tokens máximos)", 50, 400, 100, step=10)
    overlap = st.slider("Overlap (sliding window)", 0, 100, 20, step=5)
    coherence_lambda = st.slider("λ coherencia", 0.0, 2.0, 0.5, step=0.1)
    overlap_mu = st.slider("μ overlap", 0.0, 2.0, 0.3, step=0.1)
    fixed_cost = st.slider("Costo fijo por segmento", 0.0, 2000.0, 500.0, step=50.0)
    model = st.selectbox("Modelo de tokens", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"])

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

if st.button("Segmentar", type="primary"):
    with st.spinner("Calculando segmentaciones..."):
        try:
            dp_result = segment_dp(text, lmin=lmin, lmax=lmax, model=model, coherence_lambda=coherence_lambda)
            base_result = segment_baseline(text, lmax=lmax, model=model)
            sw_result = segment_sliding_window(text, lmax=lmax, overlap=overlap, model=model)
            overlap_result = segment_dp_overlap(text, lmin=lmin, lmax=lmax, model=model, coherence_lambda=coherence_lambda, overlap_mu=overlap_mu)
            report = compare(dp_result, base_result)
        except ValueError as e:
            st.error(f"Error: {e}")
            st.stop()

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
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    methods = [
        (dp_result.segments, "DP Optimal", "#378ADD"),
        (base_result.segments, "Baseline", "#EF9F27"),
        (sw_result.segments, "Sliding Window", "#1D9E75"),
        (overlap_result.segments, "DP + Overlap", "#D85A30"),
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

    tab1, tab2, tab3, tab4 = st.tabs(["DP Optimal", "Baseline", "Sliding Window", "DP + Overlap"])

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

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    st.subheader("Tabla comparativa")
    dp_m = compute_metrics(dp_result)
    base_m = compute_metrics(base_result)
    sw_m = compute_metrics(sw_result)

    # ── Curva en U — costo por número de segmentos ────────────────────────────
    st.subheader("Curva de costo por número de segmentos (DP 2D)")

    with st.spinner("Calculando DP 2D..."):
        try:
            from src.segmentation.dp import segment_dp_2d

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
    st.table({
        "Método": ["DP Optimal", "Baseline", "Sliding Window", "DP + Overlap"],
        "Segmentos": [dp_result.num_segments, base_result.num_segments, sw_result.num_segments, overlap_result.num_segments],
        "Costo total": [f"{dp_result.total_cost:,.0f}", f"{base_result.total_cost:,.0f}", f"{sw_result.total_cost:,.0f}", f"{overlap_result.total_cost:,.0f}"],
        "Avg tokens": [dp_m.avg_tokens_per_segment, base_m.avg_tokens_per_segment, sw_m.avg_tokens_per_segment, "—"],
        "Std tokens": [dp_m.std_tokens_per_segment, base_m.std_tokens_per_segment, sw_m.std_tokens_per_segment, "—"],
        "Coherencia": [dp_m.avg_coherence, base_m.avg_coherence, sw_m.avg_coherence, "—"],
    })