"""
Benchmark de comparación de métodos de segmentación.

Corre DP, Baseline, Sliding Window y DP Overlap sobre el corpus en data/corpus.json,
agrega métricas y regenera los gráficos en outputs/.

Uso:
    python3 experiments/run_benchmark.py
    python3 experiments/run_benchmark.py --lmin 20 --lmax 150
    python3 experiments/run_benchmark.py --save-json outputs/results.json
"""

import sys
import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.segmentation.dp import segment_dp, segment_dp_2d
from src.segmentation.baseline import segment_baseline
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.metrics import compute_metrics
from src.segmentation.calibration import estimate_optimal_k, validate_calibration
from src.segmentation.tokenizer import count_tokens


CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus.json",
)
OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_one(
    text: str,
    lmin: int,
    lmax: int,
    overlap: int,
    model: str,
    coherence_lambda: float,
    overlap_mu: float,
    fixed_cost: float,
) -> dict:
    dp = segment_dp(
        text, lmin=lmin, lmax=lmax, model=model,
        coherence_lambda=coherence_lambda, fixed_cost=fixed_cost,
    )
    base = segment_baseline(text, lmax=lmax, model=model)
    sw = segment_sliding_window(text, lmax=lmax, overlap=overlap, model=model)
    ov = segment_dp_overlap(
        text, lmin=lmin, lmax=lmax, model=model,
        coherence_lambda=coherence_lambda, overlap_mu=overlap_mu,
        fixed_cost=fixed_cost,
    )
    dp2d = segment_dp_2d(
        text, lmin=lmin, lmax=lmax, model=model,
        coherence_lambda=coherence_lambda, fixed_cost=fixed_cost,
    )

    m_dp = compute_metrics(dp)
    m_base = compute_metrics(base)
    m_sw = compute_metrics(sw)
    m_ov = compute_metrics(ov)

    k_star = estimate_optimal_k(text, model=model, fixed_cost=fixed_cost)
    cal = validate_calibration(
        text, lmin=lmin, lmax=lmax,
        coherence_lambda=coherence_lambda,
        fixed_cost=fixed_cost, model=model,
    )

    def pct(a, b):
        return round((b - a) / b * 100, 2) if b > 0 else 0.0

    return {
        "total_tokens": count_tokens(text, model=model),
        "dp": {
            "num_segments": m_dp.num_segments,
            "total_cost": m_dp.total_cost,
            "avg_tokens": m_dp.avg_tokens_per_segment,
            "std_tokens": m_dp.std_tokens_per_segment,
            "avg_coherence": m_dp.avg_coherence,
            "token_counts": [s.token_count for s in dp.segments],
        },
        "baseline": {
            "num_segments": m_base.num_segments,
            "total_cost": m_base.total_cost,
            "avg_tokens": m_base.avg_tokens_per_segment,
            "std_tokens": m_base.std_tokens_per_segment,
            "avg_coherence": m_base.avg_coherence,
            "token_counts": [s.token_count for s in base.segments],
        },
        "sliding_window": {
            "num_segments": sw.num_segments,
            "total_cost": sw.total_cost,
            "total_overlap_tokens": sw.total_overlap_tokens,
        },
        "dp_overlap": {
            "num_segments": ov.num_segments,
            "total_cost": ov.total_cost,
            "total_overlap_tokens": ov.total_overlap_tokens,
        },
        "dp_2d": {
            "optimal_k": dp2d.num_segments,
            "total_cost": dp2d.total_cost,
            "cost_by_k": {str(k): v for k, v in dp2d.cost_by_k.items()},
        },
        "calibration": cal,
        "k_analytical": k_star,
        "reductions": {
            "dp_vs_baseline_pct": pct(m_dp.total_cost, m_base.total_cost),
            "dp_vs_sliding_window_pct": pct(m_dp.total_cost, sw.total_cost),
            "dp_vs_overlap_pct": pct(m_dp.total_cost, m_ov.total_cost),
        },
    }


def print_results(corpus: list[dict], results: list[dict]) -> None:
    sep = "─" * 80
    print(sep)
    print(f"{'BENCHMARK — Resultados por texto':^80}")
    print(sep)

    header = f"{'Texto':<28} {'Tok':>5} {'DP seg':>6} {'DP costo':>10} {'Base costo':>10} {'DP<Base%':>9} {'DP<SW%':>8}"
    print(header)
    print("─" * 80)

    for doc, res in zip(corpus, results):
        print(
            f"{doc['title'][:27]:<28} "
            f"{res['total_tokens']:>5} "
            f"{res['dp']['num_segments']:>6} "
            f"{res['dp']['total_cost']:>10,.0f} "
            f"{res['baseline']['total_cost']:>10,.0f} "
            f"{res['reductions']['dp_vs_baseline_pct']:>8.2f}% "
            f"{res['reductions']['dp_vs_sliding_window_pct']:>7.2f}%"
        )

    print(sep)

    # promedios
    avg_vs_base = np.mean([r["reductions"]["dp_vs_baseline_pct"] for r in results])
    avg_vs_sw = np.mean([r["reductions"]["dp_vs_sliding_window_pct"] for r in results])
    avg_coh_dp = np.mean([r["dp"]["avg_coherence"] for r in results])
    avg_coh_base = np.mean([r["baseline"]["avg_coherence"] for r in results])

    print(f"\nPromedio reducción DP vs Baseline:       {avg_vs_base:.2f}%")
    print(f"Promedio reducción DP vs Sliding Window: {avg_vs_sw:.2f}%")
    print(f"Coherencia promedio DP:                  {avg_coh_dp:.4f}")
    print(f"Coherencia promedio Baseline:            {avg_coh_base:.4f}")
    print(f"Mejora coherencia DP vs Baseline:        {avg_coh_dp - avg_coh_base:+.4f}")

    print(sep)
    print("Calibración k*:")
    for doc, res in zip(corpus, results):
        cal = res["calibration"]
        valid = "✓" if cal["valid"] else "✗"
        print(
            f"  {doc['id']:<22} k_dp={cal['k_dp']:>2}  k*={cal['k_analytical']:>2}  "
            f"dev={cal['deviation_pct']:.0%}  [{valid}]"
        )
    print(sep)


def _plot_cost_vs_lmax(
    corpus: list[dict],
    lmin: int,
    model: str,
    coherence_lambda: float,
    fixed_cost: float,
    out_path: str,
) -> None:
    lmax_values = [50, 80, 100, 150, 200, 250, 300]

    # usar el primer texto como texto de referencia para la gráfica
    text = corpus[0]["text"]

    dp_costs, base_costs, sw_costs = [], [], []

    for lmax in lmax_values:
        try:
            dp_r = segment_dp(text, lmin=lmin, lmax=lmax, model=model,
                              coherence_lambda=coherence_lambda, fixed_cost=fixed_cost)
            base_r = segment_baseline(text, lmax=lmax, model=model)
            sw_r = segment_sliding_window(text, lmax=lmax, overlap=lmax // 4, model=model)
            dp_costs.append(dp_r.total_cost)
            base_costs.append(base_r.total_cost)
            sw_costs.append(sw_r.total_cost)
        except ValueError:
            dp_costs.append(None)
            base_costs.append(None)
            sw_costs.append(None)

    valid = [(lmax, dp, base, sw) for lmax, dp, base, sw
             in zip(lmax_values, dp_costs, base_costs, sw_costs)
             if dp is not None]
    xs, dp_ys, base_ys, sw_ys = zip(*valid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(xs, dp_ys, "o-", color="#378ADD", linewidth=2, label="DP Óptimo")
    ax1.plot(xs, base_ys, "s--", color="#EF9F27", linewidth=2, label="Baseline")
    ax1.plot(xs, sw_ys, "^:", color="#4CAF50", linewidth=2, label="Sliding Window")
    ax1.set_title("Costo total vs lmax")
    ax1.set_xlabel("lmax (tokens máximos por segmento)")
    ax1.set_ylabel("Costo total (tokens²)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    reductions = [
        (base - dp) / base * 100
        for dp, base in zip(dp_ys, base_ys)
    ]
    ax2.bar(xs, reductions, color="#378ADD", width=18, alpha=0.85)
    ax2.set_title("Reducción de costo DP vs Baseline (%)")
    ax2.set_xlabel("lmax (tokens máximos por segmento)")
    ax2.set_ylabel("Reducción (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis="y")
    for x, r in zip(xs, reductions):
        ax2.text(x, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f'Análisis de costo por lmax — "{corpus[0]["title"]}"', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def _plot_comparison_tokens(
    corpus: list[dict],
    results: list[dict],
    out_path: str,
) -> None:
    n = len(corpus)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {"DP Óptimo": "#378ADD", "Baseline": "#EF9F27", "Sliding Window": "#4CAF50"}

    for ax, doc, res in zip(axes, corpus, results):
        dp_counts = res["dp"]["token_counts"]
        base_counts = res["baseline"]["token_counts"]

        # normalizar al mismo número de "slots" para la visualización
        max_segs = max(len(dp_counts), len(base_counts))
        x = np.arange(max_segs)
        width = 0.35

        dp_padded = dp_counts + [0] * (max_segs - len(dp_counts))
        base_padded = base_counts + [0] * (max_segs - len(base_counts))

        ax.bar(x - width / 2, dp_padded, width, label="DP Óptimo",
               color=colors["DP Óptimo"], alpha=0.85)
        ax.bar(x + width / 2, base_padded, width, label="Baseline",
               color=colors["Baseline"], alpha=0.85)

        ax.axhline(
            y=res["dp"]["avg_tokens"], color="#378ADD",
            linestyle="--", linewidth=1.2, alpha=0.7,
        )
        ax.axhline(
            y=res["baseline"]["avg_tokens"], color="#EF9F27",
            linestyle="--", linewidth=1.2, alpha=0.7,
        )

        ax.set_title(doc["title"][:30], fontsize=9)
        ax.set_xlabel("Segmento")
        ax.set_ylabel("Tokens")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(max_segs)], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        red = res["reductions"]["dp_vs_baseline_pct"]
        ax.set_title(f"{doc['title'][:25]}\n↓{red:.1f}% costo", fontsize=8)

    plt.suptitle("Distribución de tokens por segmento: DP Óptimo vs Baseline", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark de segmentación")
    parser.add_argument("--lmin", type=int, default=20)
    parser.add_argument("--lmax", type=int, default=150)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--lambda", dest="coherence_lambda", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--fixed-cost", type=float, default=100.0)
    parser.add_argument("--save-json", type=str, default=None,
                        help="Ruta para guardar resultados en JSON")
    parser.add_argument("--no-plots", action="store_true",
                        help="Omitir regeneración de gráficos")
    args = parser.parse_args()

    corpus = load_corpus()
    print(f"Corpus cargado: {len(corpus)} textos")
    print(f"Parámetros: lmin={args.lmin}, lmax={args.lmax}, "
          f"λ={args.coherence_lambda}, μ={args.mu}, Cf={args.fixed_cost}\n")

    results = []
    for i, doc in enumerate(corpus, 1):
        print(f"[{i}/{len(corpus)}] {doc['title']}...", end=" ", flush=True)
        res = run_one(
            text=doc["text"],
            lmin=args.lmin,
            lmax=args.lmax,
            overlap=args.overlap,
            model=args.model,
            coherence_lambda=args.coherence_lambda,
            overlap_mu=args.mu,
            fixed_cost=args.fixed_cost,
        )
        results.append(res)
        print(f"✓  ({res['total_tokens']} tokens, "
              f"↓{res['reductions']['dp_vs_baseline_pct']:.1f}% vs baseline)")

    print()
    print_results(corpus, results)

    if args.save_json:
        out = {
            "params": {
                "lmin": args.lmin, "lmax": args.lmax, "overlap": args.overlap,
                "model": args.model, "coherence_lambda": args.coherence_lambda,
                "overlap_mu": args.mu, "fixed_cost": args.fixed_cost,
            },
            "corpus": [{"id": d["id"], "title": d["title"], "domain": d["domain"]}
                       for d in corpus],
            "results": results,
            "aggregates": {
                "avg_dp_vs_baseline_pct": round(
                    float(np.mean([r["reductions"]["dp_vs_baseline_pct"] for r in results])), 2),
                "avg_dp_vs_sw_pct": round(
                    float(np.mean([r["reductions"]["dp_vs_sliding_window_pct"] for r in results])), 2),
                "avg_coherence_dp": round(
                    float(np.mean([r["dp"]["avg_coherence"] for r in results])), 4),
                "avg_coherence_baseline": round(
                    float(np.mean([r["baseline"]["avg_coherence"] for r in results])), 4),
            },
        }
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nResultados guardados: {args.save_json}")

    if not args.no_plots:
        print("\nGenerando gráficos...")
        _plot_cost_vs_lmax(
            corpus=corpus,
            lmin=args.lmin,
            model=args.model,
            coherence_lambda=args.coherence_lambda,
            fixed_cost=args.fixed_cost,
            out_path=os.path.join(OUTPUTS_DIR, "cost_vs_lmax.png"),
        )
        _plot_comparison_tokens(
            corpus=corpus,
            results=results,
            out_path=os.path.join(OUTPUTS_DIR, "comparison_tokens.png"),
        )


if __name__ == "__main__":
    main()
