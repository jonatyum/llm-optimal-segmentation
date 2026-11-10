"""
Análisis estadístico completo sobre resultados del benchmark.

Carga outputs/benchmark_results.json (generado por run_benchmark.py --save-json)
y calcula para cada comparación DP vs método alternativo:
  - Media de reducción de costo con IC al 95%
  - t-test pareado
  - Cohen's d (tamaño del efecto)
  - Corrección FDR de Benjamini-Hochberg para comparaciones múltiples
  - Boxplots de distribución de reducciones

Uso:
    python experiments/statistical_analysis.py
    python experiments/statistical_analysis.py --results outputs/benchmark_results.json
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)
DEFAULT_RESULTS = os.path.join(OUTPUTS_DIR, "benchmark_results.json")


def _fdr_bh(p_values: list[float]) -> list[float]:
    """Corrección FDR de Benjamini-Hochberg con enforcement de monotonicidad."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [p_values[order[i]] * n / (i + 1) for i in range(n)]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = [0.0] * n
    for rank, idx in enumerate(order):
        result[idx] = min(adjusted[rank], 1.0)
    return result


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d para muestras pareadas: d = mean(b-a) / std(b-a, ddof=1)."""
    diff = b - a
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def _effect_label(d: float) -> str:
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "grande"
    if abs_d >= 0.5:
        return "medio"
    if abs_d >= 0.2:
        return "pequeño"
    return "despreciable"


def analyze_comparison(
    label: str,
    dp_costs: np.ndarray,
    other_costs: np.ndarray,
) -> dict:
    """Estadísticas completas para una comparación DP vs método alternativo."""
    n = len(dp_costs)
    # Reducción porcentual: positivo = DP es mejor (menor costo)
    reductions = (other_costs - dp_costs) / np.where(other_costs > 0, other_costs, 1.0) * 100
    mean_red = float(np.mean(reductions))
    std_red = float(np.std(reductions, ddof=1))
    sem = scipy.stats.sem(reductions)
    if sem == 0 or np.isnan(sem):
        ci = (mean_red, mean_red)
        t_stat, p_value = 0.0, 1.0
    else:
        ci = scipy.stats.t.interval(0.95, df=n - 1, loc=mean_red, scale=sem)
        t_stat, p_value = scipy.stats.ttest_rel(other_costs, dp_costs)
    d = cohens_d_paired(dp_costs, other_costs)

    return {
        "comparison": label,
        "n": n,
        "mean_reduction_pct": round(mean_red, 4),
        "std_reduction_pct": round(std_red, 4),
        "ci95_low": round(float(ci[0]), 4),
        "ci95_high": round(float(ci[1]), 4),
        "t_stat": round(float(t_stat), 6),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(d, 4),
        "effect_size": _effect_label(d),
        "significant": bool(p_value < 0.05),
        "reductions": reductions.tolist(),
    }


def run_analysis(results: list[dict]) -> list[dict]:
    # full_cost disponible desde benchmarks recientes; fallback a total_cost
    dp_full    = np.array([r["dp"].get("full_cost", r["dp"]["total_cost"]) for r in results])
    base_full  = np.array([r["baseline"].get("full_cost", r["baseline"]["total_cost"]) for r in results])
    dp_native  = np.array([r["dp"]["total_cost"] for r in results])
    sw_costs   = np.array([r["sliding_window"]["total_cost"] for r in results
                           if "sliding_window" in r])
    ov_costs   = np.array([r["dp_overlap"]["total_cost"] for r in results
                           if "dp_overlap" in r])
    tt_costs   = np.array([r["texttiling"]["total_cost"] for r in results
                           if "texttiling" in r])

    comparisons = [
        analyze_comparison("DP vs Baseline", dp_full, base_full),
    ]
    if len(sw_costs) == len(dp_native):
        comparisons.append(analyze_comparison("DP vs Sliding Window", dp_native, sw_costs))
    if len(ov_costs) == len(dp_native):
        comparisons.append(analyze_comparison("DP vs DP+Overlap", dp_native, ov_costs))
    if len(tt_costs) == len(dp_native):
        comparisons.append(analyze_comparison("DP vs TextTiling", dp_native, tt_costs))

    # FDR sobre las 4 comparaciones
    p_values = [c["p_value"] for c in comparisons]
    fdr_corrected = _fdr_bh(p_values)
    for comp, p_fdr in zip(comparisons, fdr_corrected):
        comp["p_value_fdr"] = round(p_fdr, 6)
        comp["significant_fdr"] = bool(p_fdr < 0.05)

    return comparisons


def print_analysis(comparisons: list[dict]) -> None:
    sep = "═" * 100
    thin = "─" * 100
    print(sep)
    print(f"{'ANÁLISIS ESTADÍSTICO — DP Óptimo vs Métodos Alternativos':^100}")
    print(sep)
    print(
        f"  {'Comparación':<28} {'Reduc.%':>8} {'IC 95%':>20} "
        f"{'t':>9} {'p':>10} {'p(FDR)':>10} {'d':>7} {'Efecto':<12} {'Sig.':>5}"
    )
    print("  " + thin)

    for c in comparisons:
        sig = "✓" if c["significant"] else "✗"
        fdr_sig = "✓" if c["significant_fdr"] else "✗"
        print(
            f"  {c['comparison']:<28} "
            f"{c['mean_reduction_pct']:>7.2f}% "
            f"[{c['ci95_low']:>6.2f}%,{c['ci95_high']:>6.2f}%] "
            f"{c['t_stat']:>9.4f} "
            f"{c['p_value']:>10.4f} "
            f"{c['p_value_fdr']:>10.4f} "
            f"{c['cohens_d']:>7.4f} "
            f"{c['effect_size']:<12} "
            f"{sig}/{fdr_sig}"
        )

    print("  " + thin)
    print("  Sig. = significativo (α=0.05) sin corr. / con FDR (Benjamini-Hochberg)")
    print("  Reducción: % de reducción de costo; positivo = DP es mejor")
    print("  Cohen's d: |d|≥0.8 grande, ≥0.5 medio, ≥0.2 pequeño, <0.2 despreciable")
    print(sep)

    print("\nDetalle por comparación:\n")
    for c in comparisons:
        print(f"  {c['comparison']}  (n={c['n']} textos):")
        print(f"    Reducción media:   {c['mean_reduction_pct']:.2f}% ± {c['std_reduction_pct']:.2f}%")
        print(f"    IC 95%:            [{c['ci95_low']:.2f}%, {c['ci95_high']:.2f}%]")
        print(f"    t({c['n']-1}) = {c['t_stat']:.4f},  p = {c['p_value']:.4f},  p_FDR = {c['p_value_fdr']:.4f}")
        print(f"    Cohen's d = {c['cohens_d']:.4f}  ({c['effect_size']})")
        print(f"    Significativo (α=0.05): {'Sí' if c['significant'] else 'No'}  |  Sig. FDR: {'Sí' if c['significant_fdr'] else 'No'}")
        print()


def _plot_boxplots(comparisons: list[dict], out_path: str) -> None:
    colors = ["#378ADD", "#EF9F27", "#1D9E75", "#7B5EA7"]
    fig, ax = plt.subplots(figsize=(13, 7))

    data = [c["reductions"] for c in comparisons]
    labels = [c["comparison"].replace("DP vs ", "") for c in comparisons]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
        whiskerprops=dict(linestyle="--", linewidth=1.2, color="#555"),
        capprops=dict(linewidth=1.2, color="#555"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_color(color)

    # Scatter de puntos individuales
    for i, (comp, color) in enumerate(zip(comparisons, colors), 1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(comp["reductions"]))
        ax.scatter(
            np.full(len(comp["reductions"]), i) + jitter,
            comp["reductions"],
            color=color, alpha=0.55, s=30, zorder=3,
        )

    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.75)

    # Anotaciones d y p — justo encima del cap superior de cada caja
    whisker_tops = [max(w.get_ydata()) for w in bp["whiskers"][1::2]]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, (comp, top) in enumerate(zip(comparisons, whisker_tops), 1):
        sig_text = f"d = {comp['cohens_d']:.2f}\np = {comp['p_value']:.3f}"
        ax.text(i, top + y_range * 0.03, sig_text,
                ha="center", va="bottom", fontsize=9, color="#222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    ax.set_ylabel("Reducción de costo vs DP (%)", fontsize=12)
    ax.set_title(
        "DP Óptimo: reducción de costo vs métodos alternativos\n"
        "Cada punto = un texto del corpus · Caja = IQR · Línea blanca = mediana",
        fontsize=12, pad=14,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color=c, label=f"{comp['comparison']} — {comp['effect_size']} (d={comp['cohens_d']:.2f})")
        for comp, c in zip(comparisons, colors)
    ]
    ax.legend(
        handles=patches + [plt.Line2D([0], [0], color="#d62728", linestyle="--",
                                       linewidth=1.4, label="Sin reducción")],
        fontsize=9, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Análisis estadístico del benchmark")
    parser.add_argument(
        "--results", type=str, default=DEFAULT_RESULTS,
        help="Ruta al JSON de resultados del benchmark"
    )
    parser.add_argument("--no-plots", action="store_true", help="Omitir figuras")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: No se encuentra {args.results}")
        print("Ejecuta primero:")
        print("  python experiments/run_benchmark.py --save-json outputs/benchmark_results.json")
        sys.exit(1)

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    print(f"Cargados {len(results)} textos desde {args.results}\n")

    comparisons = run_analysis(results)
    print_analysis(comparisons)

    out_json = os.path.join(OUTPUTS_DIR, "statistical_results.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_texts": len(results), "comparisons": comparisons}, f,
                  indent=2, ensure_ascii=False)
    print(f"Resultados guardados: {out_json}")

    if not args.no_plots:
        _plot_boxplots(comparisons, os.path.join(OUTPUTS_DIR, "statistical_analysis.png"))


if __name__ == "__main__":
    main()
