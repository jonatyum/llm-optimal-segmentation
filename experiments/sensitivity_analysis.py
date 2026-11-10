"""
Análisis de sensibilidad: grid de parámetros λ × Cf sobre el corpus.

Para cada combinación (λ, Cf) corre DP y Baseline sobre un subconjunto
representativo del corpus y registra:
  - k* promedio (número óptimo de segmentos)
  - Reducción de costo media DP vs Baseline (%)

Genera dos heatmaps y guarda los resultados en JSON.

Uso:
    python experiments/sensitivity_analysis.py
    python experiments/sensitivity_analysis.py --lmin 20 --lmax 150 --n-texts 5
    python experiments/sensitivity_analysis.py --all-texts     (usa los 20 textos — lento)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.segmentation.dp import segment_dp
from src.segmentation.baseline import segment_baseline
from src.segmentation.metrics import recompute_full_cost

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus.json",
)
OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)

LAMBDA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
CF_VALUES = [200.0, 500.0, 1000.0, 2000.0, 5000.0]


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def select_representative_texts(corpus: list[dict], n: int) -> list[dict]:
    """Selecciona n textos representativos distribuidos uniformemente por longitud."""
    if n >= len(corpus):
        return corpus
    indices = np.round(np.linspace(0, len(corpus) - 1, n)).astype(int)
    return [corpus[i] for i in indices]


def run_grid(
    texts: list[dict],
    lmin: int,
    lmax: int,
    model: str,
    max_sentences: int,
) -> dict:
    """Ejecuta el grid completo λ × Cf sobre los textos dados."""
    results = {}
    total = len(LAMBDA_VALUES) * len(CF_VALUES) * len(texts)
    done = 0

    for lam in LAMBDA_VALUES:
        for cf in CF_VALUES:
            key = (lam, cf)
            k_stars, reductions = [], []

            for doc in texts:
                text = doc["text"]
                try:
                    dp = segment_dp(
                        text, lmin=lmin, lmax=lmax, model=model,
                        coherence_lambda=lam, fixed_cost=cf,
                        max_sentences=max_sentences,
                    )
                    base = segment_baseline(text, lmin=lmin, lmax=lmax, model=model,
                                            max_sentences=max_sentences)
                    dp_cost = recompute_full_cost(dp, coherence_lambda=lam, fixed_cost=cf, model=model)
                    base_cost = recompute_full_cost(base, coherence_lambda=lam, fixed_cost=cf, model=model)
                    k_stars.append(dp.num_segments)
                    red = (base_cost - dp_cost) / base_cost * 100 if base_cost > 0 else 0.0
                    reductions.append(red)
                except Exception:
                    pass

                done += 1
                pct = done / total * 100
                print(f"\r  [{pct:5.1f}%] λ={lam:.1f}, Cf={cf:.0f}, texto={doc['id']}    ", end="", flush=True)

            results[key] = {
                "lambda": lam,
                "fixed_cost": cf,
                "mean_k_star": float(np.mean(k_stars)) if k_stars else float("nan"),
                "mean_reduction_pct": float(np.mean(reductions)) if reductions else float("nan"),
                "n_texts": len(k_stars),
            }

    print()
    return results


def _build_matrices(results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Construye matrices 2D de k* y reducción indexadas por (λ, Cf)."""
    n_lam = len(LAMBDA_VALUES)
    n_cf = len(CF_VALUES)
    k_matrix = np.full((n_cf, n_lam), float("nan"))
    red_matrix = np.full((n_cf, n_lam), float("nan"))

    for (lam, cf), vals in results.items():
        i = CF_VALUES.index(cf)
        j = LAMBDA_VALUES.index(lam)
        k_matrix[i, j] = vals["mean_k_star"]
        red_matrix[i, j] = vals["mean_reduction_pct"]

    return k_matrix, red_matrix


def _annotate_heatmap(ax, matrix: np.ndarray, fmt: str = ".1f") -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=9, fontweight="600",
                        color="white" if abs(val) > 0.6 * np.nanmax(np.abs(matrix)) else "#111")


def _plot_heatmaps(k_matrix: np.ndarray, red_matrix: np.ndarray, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lam_labels = [str(v) for v in LAMBDA_VALUES]
    cf_labels = [f"{int(v):,}" for v in CF_VALUES]

    # Heatmap k*
    im1 = ax1.imshow(k_matrix, cmap="YlOrRd", aspect="auto",
                     vmin=np.nanmin(k_matrix), vmax=np.nanmax(k_matrix))
    ax1.set_xticks(range(len(LAMBDA_VALUES)))
    ax1.set_xticklabels(lam_labels)
    ax1.set_yticks(range(len(CF_VALUES)))
    ax1.set_yticklabels(cf_labels)
    ax1.set_xlabel("λ (peso de coherencia)", fontsize=11)
    ax1.set_ylabel("Cf (costo fijo por segmento)", fontsize=11)
    ax1.set_title("k* promedio (número óptimo de segmentos)", fontsize=12)
    plt.colorbar(im1, ax=ax1, label="k* promedio")
    _annotate_heatmap(ax1, k_matrix, ".1f")

    # Heatmap reducción de costo
    vmin = np.nanmin(red_matrix)
    vmax = np.nanmax(red_matrix)
    vcenter = (vmin + vmax) / 2
    if vmin < vcenter < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None
    im2 = ax2.imshow(red_matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    ax2.set_xticks(range(len(LAMBDA_VALUES)))
    ax2.set_xticklabels(lam_labels)
    ax2.set_yticks(range(len(CF_VALUES)))
    ax2.set_yticklabels(cf_labels)
    ax2.set_xlabel("λ (peso de coherencia)", fontsize=11)
    ax2.set_ylabel("Cf (costo fijo por segmento)", fontsize=11)
    ax2.set_title("Reducción de costo DP vs Baseline (%)", fontsize=12)
    plt.colorbar(im2, ax=ax2, label="Reducción promedio (%)")
    _annotate_heatmap(ax2, red_matrix, ".1f")

    plt.suptitle(
        "Análisis de sensibilidad: efecto de λ y Cf sobre la segmentación óptima",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def print_grid_summary(results: dict) -> None:
    sep = "─" * 72
    print(sep)
    print(f"{'ANÁLISIS DE SENSIBILIDAD — Grid λ × Cf':^72}")
    print(sep)
    print(f"  {'λ':>5} {'Cf':>7}  {'k* prom':>9}  {'Reduc%':>9}  n")
    print("  " + sep)
    for (lam, cf), v in sorted(results.items()):
        k = v["mean_k_star"]
        r = v["mean_reduction_pct"]
        print(
            f"  {lam:>5.1f} {cf:>7.0f}  "
            f"{k:>9.2f}  {r:>8.2f}%  {v['n_texts']}"
        )
    print(sep)

    # Resumen de variabilidad
    all_k = [v["mean_k_star"] for v in results.values() if not np.isnan(v["mean_k_star"])]
    all_r = [v["mean_reduction_pct"] for v in results.values() if not np.isnan(v["mean_reduction_pct"])]
    print(f"\nk*:          rango [{min(all_k):.1f}, {max(all_k):.1f}]  —  variación = {max(all_k)-min(all_k):.1f} segmentos")
    print(f"Reducción%:  rango [{min(all_r):.1f}%, {max(all_r):.1f}%]  —  variación = {max(all_r)-min(all_r):.1f}pp")


def main():
    parser = argparse.ArgumentParser(description="Análisis de sensibilidad λ × Cf")
    parser.add_argument("--lmin", type=int, default=20)
    parser.add_argument("--lmax", type=int, default=150)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max-sentences", type=int, default=100,
                        help="Límite de oraciones para el DP (más = más lento)")
    parser.add_argument("--n-texts", type=int, default=5,
                        help="Número de textos del corpus a usar (por defecto: 5)")
    parser.add_argument("--all-texts", action="store_true",
                        help="Usar los 20 textos del corpus (lento)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    corpus = load_corpus()
    if args.all_texts:
        texts = corpus
    else:
        texts = select_representative_texts(corpus, args.n_texts)

    print(f"Corpus: {len(texts)} textos seleccionados de {len(corpus)}")
    print(f"Grid: {len(LAMBDA_VALUES)} valores λ × {len(CF_VALUES)} valores Cf = "
          f"{len(LAMBDA_VALUES)*len(CF_VALUES)} combinaciones")
    print(f"Total runs: {len(LAMBDA_VALUES)*len(CF_VALUES)*len(texts)}\n")

    t0 = time.perf_counter()
    results = run_grid(texts, args.lmin, args.lmax, args.model, args.max_sentences)
    elapsed = time.perf_counter() - t0
    print(f"Tiempo total: {elapsed:.1f}s\n")

    print_grid_summary(results)

    # Guardar resultados
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUTS_DIR, "sensitivity_results.json")
    serializable = {
        f"lambda={lam}_cf={cf}": v
        for (lam, cf), v in results.items()
    }
    meta = {
        "params": {
            "lmin": args.lmin, "lmax": args.lmax, "model": args.model,
            "max_sentences": args.max_sentences, "n_texts": len(texts),
        },
        "lambda_values": LAMBDA_VALUES,
        "cf_values": CF_VALUES,
        "results": serializable,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados: {out_json}")

    if not args.no_plots:
        k_matrix, red_matrix = _build_matrices(results)
        _plot_heatmaps(k_matrix, red_matrix, os.path.join(OUTPUTS_DIR, "sensitivity_analysis.png"))


if __name__ == "__main__":
    main()
