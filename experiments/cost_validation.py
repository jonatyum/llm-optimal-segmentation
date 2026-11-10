"""
Validación del modelo de costo teórico contra latencia real de Ollama.

El modelo teórico afirma que el costo de procesar un segmento es:
    C(t, coh) = t² · (1 + λ·(1 − coh)) + Cf

donde t = tokens del segmento, coh = coherencia semántica interna.

Este script:
  1. Segmenta textos del corpus con DP y Baseline
  2. Corre inferencia real con Ollama (gemma2:2b) en cada segmento
  3. Compara: costo teórico predicho vs latencia real medida
  4. Calcula R², correlación de Pearson y RMSE
  5. Genera scatter plots con línea de regresión

Hipótesis: latencia ≈ α · t² + β (relación cuadrática con los tokens)
Validación: R²(t², latency) debe ser > 0.7 para confirmar el modelo.

Requiere Ollama corriendo localmente con gemma2:2b descargado:
    ollama pull gemma2:2b

Uso:
    python experiments/cost_validation.py
    python experiments/cost_validation.py --n-texts 3 --model gemma2:2b
    python experiments/cost_validation.py --no-plots --save-json outputs/cost_validation.json
"""

import sys
import os
import json
import argparse
import time

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.segmentation.dp import segment_dp
from src.segmentation.baseline import segment_baseline
from src.segmentation.metrics import compute_metrics
from src.segmentation.embeddings import get_embeddings, cosine_similarity, clear_embedding_cache
from src.segmentation.splitter import split_sentences
from src.segmentation.tokenizer import count_tokens

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus.json",
)
OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)


def _check_ollama(model: str) -> bool:
    """Verifica que Ollama esté disponible y el modelo descargado."""
    try:
        import ollama
        ollama.show(model)
        return True
    except Exception:
        return False


def _run_inference_timed(segment_text: str, model: str) -> float | None:
    """Corre inferencia real y retorna latencia en ms. None si falla."""
    try:
        import ollama
        t0 = time.perf_counter()
        ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following text in one sentence."},
                {"role": "user", "content": segment_text},
            ],
        )
        return (time.perf_counter() - t0) * 1000
    except Exception:
        return None


def _segment_coherence(sentences: list[str], start: int, end: int) -> float:
    """Coherencia interna del segmento: media de similitudes coseno consecutivas."""
    seg_sents = sentences[start:end]
    if len(seg_sents) < 2:
        return 1.0
    embeddings = get_embeddings(seg_sents)
    sims = [cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)]
    return float(np.mean(sims))


def _theoretical_cost(token_count: int, coherence: float,
                       coherence_lambda: float, fixed_cost: float) -> float:
    return float(token_count ** 2) * (1 + coherence_lambda * (1 - coherence)) + fixed_cost


def collect_data(
    corpus: list[dict],
    n_texts: int,
    lmin: int,
    lmax: int,
    coherence_lambda: float,
    fixed_cost: float,
    model: str,
    llm_model: str,
) -> list[dict]:
    """Corre DP y Baseline en n_texts, luego inferencia real en cada segmento."""
    texts = corpus[:n_texts]
    records = []

    for doc in texts:
        text = doc["text"]
        sentences = split_sentences(text)
        embeddings = get_embeddings(sentences)
        print(f"  {doc['id']} ({len(sentences)} oraciones)...")

        for method_name, result in [
            ("dp", segment_dp(text, lmin=lmin, lmax=lmax, model=model,
                              coherence_lambda=coherence_lambda, fixed_cost=fixed_cost)),
            ("baseline", segment_baseline(text, lmin=lmin, lmax=lmax, model=model)),
        ]:
            offset = 0
            for seg in result.segments:
                n = len(seg.sentences)
                start, end = offset, offset + n
                coh = _segment_coherence(sentences, start, end)
                t_count = seg.token_count
                pred_cost = _theoretical_cost(t_count, coh, coherence_lambda, fixed_cost)

                print(f"    {method_name} seg {seg.index}: {t_count} tok, coh={coh:.3f}, "
                      f"pred={pred_cost:,.0f} → inferencia...", end=" ", flush=True)

                latency = _run_inference_timed(seg.text, llm_model)
                if latency is not None:
                    print(f"{latency:.0f}ms")
                    records.append({
                        "doc_id": doc["id"],
                        "method": method_name,
                        "segment_idx": seg.index,
                        "token_count": t_count,
                        "token_count_sq": t_count ** 2,
                        "coherence": round(coh, 4),
                        "predicted_cost": round(pred_cost, 2),
                        "latency_ms": round(latency, 2),
                    })
                else:
                    print("FALLO")

                offset += n

        clear_embedding_cache()

    return records


def compute_validation_stats(records: list[dict]) -> dict:
    """Regresión lineal y correlaciones entre costo predicho y latencia real."""
    if len(records) < 3:
        return {}

    t_sq = np.array([r["token_count_sq"] for r in records], dtype=float)
    pred = np.array([r["predicted_cost"] for r in records], dtype=float)
    lat = np.array([r["latency_ms"] for r in records], dtype=float)

    def _regress(x, y):
        slope, intercept, r, p, se = scipy.stats.linregress(x, y)
        return {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 2),
            "r": round(float(r), 4),
            "r2": round(float(r ** 2), 4),
            "p_value": round(float(p), 6),
            "std_err": round(float(se), 6),
        }

    reg_tsq = _regress(t_sq, lat)
    reg_pred = _regress(pred, lat)

    # RMSE relativo
    y_hat_tsq = reg_tsq["slope"] * t_sq + reg_tsq["intercept"]
    y_hat_pred = reg_pred["slope"] * pred + reg_pred["intercept"]
    rmse_tsq = float(np.sqrt(np.mean((lat - y_hat_tsq) ** 2)))
    rmse_pred = float(np.sqrt(np.mean((lat - y_hat_pred) ** 2)))

    return {
        "n_segments": len(records),
        "latency_mean_ms": round(float(np.mean(lat)), 2),
        "latency_std_ms": round(float(np.std(lat)), 2),
        "regression_t2_vs_latency": {**reg_tsq, "rmse_ms": round(rmse_tsq, 2)},
        "regression_predicted_vs_latency": {**reg_pred, "rmse_ms": round(rmse_pred, 2)},
    }


def print_validation(stats: dict) -> None:
    if not stats:
        print("Insuficientes datos para calcular estadísticas.")
        return
    sep = "─" * 72
    print(sep)
    print(f"{'VALIDACIÓN DEL MODELO DE COSTO vs LATENCIA REAL':^72}")
    print(sep)
    print(f"  n segmentos medidos: {stats['n_segments']}")
    print(f"  Latencia real:  media = {stats['latency_mean_ms']:.1f}ms  "
          f"± {stats['latency_std_ms']:.1f}ms")
    print()

    r1 = stats["regression_t2_vs_latency"]
    r2 = stats["regression_predicted_vs_latency"]

    print(f"  Regresión: latency = α·t² + β")
    print(f"    α = {r1['slope']:.6f},  β = {r1['intercept']:.1f}")
    print(f"    R² = {r1['r2']:.4f},  r = {r1['r']:.4f},  p = {r1['p_value']:.4f}")
    print(f"    RMSE = {r1['rmse_ms']:.1f}ms")
    print()
    print(f"  Regresión: latency = α·C_pred + β  (C_pred incluye λ y coherencia)")
    print(f"    α = {r2['slope']:.6f},  β = {r2['intercept']:.1f}")
    print(f"    R² = {r2['r2']:.4f},  r = {r2['r']:.4f},  p = {r2['p_value']:.4f}")
    print(f"    RMSE = {r2['rmse_ms']:.1f}ms")
    print()

    validated = r1["r2"] > 0.5
    verdict = (
        f"VALIDADO — R²(t², latency)={r1['r2']:.3f} > 0.5: "
        "el modelo cuadrático predice bien la latencia"
        if validated
        else f"PARCIAL — R²(t², latency)={r1['r2']:.3f}: "
        "la relación cuadrática existe pero con alta varianza"
    )
    print(f"  Veredicto: {verdict}")
    print(sep)


def _plot_validation(records: list[dict], stats: dict, out_path: str) -> None:
    if not records or not stats:
        return

    t_sq = np.array([r["token_count_sq"] for r in records], dtype=float)
    pred = np.array([r["predicted_cost"] for r in records], dtype=float)
    lat = np.array([r["latency_ms"] for r in records], dtype=float)
    methods = [r["method"] for r in records]

    colors = ["#378ADD" if m == "dp" else "#EF9F27" for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: t² vs latency
    ax1.scatter(t_sq, lat, c=colors, s=60, alpha=0.75, zorder=3)
    reg = stats["regression_t2_vs_latency"]
    x_line = np.linspace(t_sq.min(), t_sq.max(), 100)
    ax1.plot(x_line, reg["slope"] * x_line + reg["intercept"],
             "k-", linewidth=2, alpha=0.7,
             label=f"Regresión lineal\nR²={reg['r2']:.3f}  RMSE={reg['rmse_ms']:.0f}ms")
    ax1.set_xlabel("t² (tokens²)", fontsize=11)
    ax1.set_ylabel("Latencia real (ms)", fontsize=11)
    ax1.set_title("Latencia real vs t² (costo cuadrático puro)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    handles = [
        mpatches.Patch(color="#378ADD", label="DP Óptimo"),
        mpatches.Patch(color="#EF9F27", label="Baseline"),
    ]
    ax1.legend(handles=handles + [
        plt.Line2D([0], [0], color="k", linewidth=2,
                   label=f"Regresión  R²={reg['r2']:.3f}")
    ], fontsize=9)

    # Plot 2: predicted cost vs latency
    ax2.scatter(pred, lat, c=colors, s=60, alpha=0.75, zorder=3)
    reg2 = stats["regression_predicted_vs_latency"]
    x_line2 = np.linspace(pred.min(), pred.max(), 100)
    ax2.plot(x_line2, reg2["slope"] * x_line2 + reg2["intercept"],
             "k-", linewidth=2, alpha=0.7)
    ax2.set_xlabel("Costo predicho C(t,coh) = t²·(1+λ(1−coh))+Cf", fontsize=10)
    ax2.set_ylabel("Latencia real (ms)", fontsize=11)
    ax2.set_title("Latencia real vs costo predicho completo", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=handles + [
        plt.Line2D([0], [0], color="k", linewidth=2,
                   label=f"Regresión  R²={reg2['r2']:.3f}")
    ], fontsize=9)

    plt.suptitle(
        "Validación del modelo de costo: predicción teórica vs latencia Ollama real",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validación del modelo de costo con Ollama")
    parser.add_argument("--n-texts", type=int, default=3,
                        help="Número de textos del corpus a usar (default: 3)")
    parser.add_argument("--llm-model", type=str, default="gemma2:2b",
                        help="Modelo Ollama para inferencia real")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Tokenizador para contar tokens")
    parser.add_argument("--lmin", type=int, default=20)
    parser.add_argument("--lmax", type=int, default=150)
    parser.add_argument("--lambda", dest="coherence_lambda", type=float, default=0.5)
    parser.add_argument("--fixed-cost", type=float, default=1000.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    print(f"Verificando Ollama ({args.llm_model})...", end=" ")
    if not _check_ollama(args.llm_model):
        print("NO DISPONIBLE")
        print(f"\nERROR: Ollama no está corriendo o el modelo '{args.llm_model}' no está descargado.")
        print("Para instalar:")
        print("  1. Descarga Ollama: https://ollama.com/download")
        print(f"  2. ollama pull {args.llm_model}")
        print("  3. Vuelve a correr este script")
        sys.exit(1)
    print("OK")

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"\nCorriendo inferencia real en {args.n_texts} textos × 2 métodos...\n")
    records = collect_data(
        corpus=corpus,
        n_texts=args.n_texts,
        lmin=args.lmin,
        lmax=args.lmax,
        coherence_lambda=args.coherence_lambda,
        fixed_cost=args.fixed_cost,
        model=args.model,
        llm_model=args.llm_model,
    )

    if not records:
        print("\nNo se obtuvieron registros. Verifica que Ollama esté respondiendo.")
        sys.exit(1)

    print(f"\nTotal segmentos medidos: {len(records)}\n")
    stats = compute_validation_stats(records)
    print_validation(stats)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = args.save_json or os.path.join(OUTPUTS_DIR, "cost_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"records": records, "stats": stats}, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados: {out_path}")

    if not args.no_plots and records:
        _plot_validation(records, stats, os.path.join(OUTPUTS_DIR, "cost_validation.png"))


if __name__ == "__main__":
    main()
