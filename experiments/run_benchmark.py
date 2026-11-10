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
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.segmentation.dp import segment_dp, segment_dp_2d
from src.segmentation.baseline import segment_baseline
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.texttiling import segment_texttiling
from src.segmentation.metrics import compute_metrics, recompute_full_cost
from src.segmentation.calibration import estimate_optimal_k, validate_calibration
from src.segmentation.tokenizer import count_tokens
from src.segmentation.segmentation_metrics import boundaries_from_result, pk, window_diff


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
    tt = segment_texttiling(text, lmin=lmin, lmax=lmax, model=model)

    m_dp = compute_metrics(dp)
    m_base = compute_metrics(base)
    m_sw = compute_metrics(sw)
    m_ov = compute_metrics(ov)
    m_tt = compute_metrics(tt)

    # FIX 12: usar recompute_full_cost para comparación justa con la misma función de costo
    dp_full_cost = recompute_full_cost(
        dp, coherence_lambda=coherence_lambda, fixed_cost=fixed_cost, model=model
    )
    base_full_cost = recompute_full_cost(
        base, coherence_lambda=coherence_lambda, fixed_cost=fixed_cost, model=model
    )

    k_star = estimate_optimal_k(
        text, model=model, fixed_cost=fixed_cost, coherence_lambda=coherence_lambda
    )
    cal = validate_calibration(
        text, lmin=lmin, lmax=lmax,
        coherence_lambda=coherence_lambda,
        fixed_cost=fixed_cost, model=model,
    )

    def pct(a, b):
        return round((b - a) / b * 100, 2) if b > 0 else 0.0

    # Métricas de frontera Pk y WindowDiff — DP como referencia óptima
    try:
        dp_bounds = boundaries_from_result(dp)
        _pk_base = pk(dp_bounds, boundaries_from_result(base))
        _pk_sw = pk(dp_bounds, boundaries_from_result(sw))
        _pk_ov = pk(dp_bounds, boundaries_from_result(ov))
        _pk_tt = pk(dp_bounds, boundaries_from_result(tt))
        _wd_base = window_diff(dp_bounds, boundaries_from_result(base))
        _wd_sw = window_diff(dp_bounds, boundaries_from_result(sw))
        _wd_ov = window_diff(dp_bounds, boundaries_from_result(ov))
        _wd_tt = window_diff(dp_bounds, boundaries_from_result(tt))
        boundary_metrics = {
            "baseline":       {"pk_vs_dp": round(_pk_base, 6), "wd_vs_dp": round(_wd_base, 6)},
            "sliding_window": {"pk_vs_dp": round(_pk_sw, 6),   "wd_vs_dp": round(_wd_sw, 6)},
            "dp_overlap":     {"pk_vs_dp": round(_pk_ov, 6),   "wd_vs_dp": round(_wd_ov, 6)},
            "texttiling":     {"pk_vs_dp": round(_pk_tt, 6),   "wd_vs_dp": round(_wd_tt, 6)},
        }
    except Exception:
        boundary_metrics = {}

    return {
        "total_tokens": count_tokens(text, model=model),
        "dp": {
            "num_segments": m_dp.num_segments,
            "total_cost": m_dp.total_cost,
            "full_cost": dp_full_cost,
            "avg_tokens": m_dp.avg_tokens_per_segment,
            "std_tokens": m_dp.std_tokens_per_segment,
            "avg_coherence": m_dp.avg_coherence,
            "token_counts": [s.token_count for s in dp.segments],
        },
        "baseline": {
            "num_segments": m_base.num_segments,
            "total_cost": m_base.total_cost,
            "full_cost": base_full_cost,
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
        "texttiling": {
            "num_segments": m_tt.num_segments,
            "total_cost": m_tt.total_cost,
            "avg_tokens": m_tt.avg_tokens_per_segment,
            "std_tokens": m_tt.std_tokens_per_segment,
            "avg_coherence": m_tt.avg_coherence,
            "token_counts": [s.token_count for s in tt.segments],
        },
        "calibration": cal,
        "k_analytical": k_star,
        "boundary_metrics": boundary_metrics,
        "reductions": {
            "dp_vs_baseline_pct": pct(m_dp.total_cost, m_base.total_cost),
            "dp_vs_baseline_full_pct": pct(dp_full_cost, base_full_cost),
            "dp_vs_sliding_window_pct": pct(m_dp.total_cost, sw.total_cost),
            "dp_vs_overlap_pct": pct(m_dp.total_cost, m_ov.total_cost),
            "dp_vs_texttiling_pct": pct(m_dp.total_cost, m_tt.total_cost),
        },
    }


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d para muestras pareadas: d = mean(b-a) / std(b-a, ddof=1)."""
    diff = b - a
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


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


def _analyze_pair(label: str, dp_c: np.ndarray, other_c: np.ndarray) -> dict:
    n = len(dp_c)
    reductions = (other_c - dp_c) / np.where(other_c > 0, other_c, 1.0) * 100
    mean_red = float(np.mean(reductions))
    ci = scipy.stats.t.interval(0.95, df=n - 1, loc=mean_red, scale=scipy.stats.sem(reductions))
    t_stat, p_value = scipy.stats.ttest_rel(other_c, dp_c)
    d = _cohens_d_paired(dp_c, other_c)
    return {
        "comparison": label,
        "n": n,
        "mean_reduction_pct": round(mean_red, 4),
        "ci95_low": round(float(ci[0]), 4),
        "ci95_high": round(float(ci[1]), 4),
        "t_stat": round(float(t_stat), 6),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(d, 4),
        "significant": bool(p_value < 0.05),
    }


def _compute_statistics(results: list[dict]) -> dict:
    """Tests estadísticos completos: t-test, Cohen's d, IC95%, FDR para todas las comparaciones."""
    if len(results) < 2:
        return {}

    dp_full = np.array([r["dp"]["full_cost"] for r in results])
    base_full = np.array([r["baseline"]["full_cost"] for r in results])
    dp_native = np.array([r["dp"]["total_cost"] for r in results])
    sw_costs = np.array([r["sliding_window"]["total_cost"] for r in results])
    ov_costs = np.array([r["dp_overlap"]["total_cost"] for r in results])
    tt_costs = np.array([r["texttiling"]["total_cost"] for r in results])

    comparisons = [
        _analyze_pair("DP vs Baseline (full cost)", dp_full, base_full),
        _analyze_pair("DP vs Sliding Window", dp_native, sw_costs),
        _analyze_pair("DP vs DP+Overlap", dp_native, ov_costs),
        _analyze_pair("DP vs TextTiling", dp_native, tt_costs),
    ]

    # FDR sobre las 4 comparaciones (Benjamini-Hochberg)
    fdr = _fdr_bh([c["p_value"] for c in comparisons])
    for comp, p_adj in zip(comparisons, fdr):
        comp["p_value_fdr"] = round(p_adj, 6)
        comp["significant_fdr"] = bool(p_adj < 0.05)

    return {"comparisons": comparisons}


def print_results(corpus: list[dict], results: list[dict]) -> None:
    sep = "─" * 88
    print(sep)
    print(f"{'BENCHMARK — Resultados por texto':^88}")
    print(sep)

    header = f"{'Texto':<28} {'Tok':>5} {'DP seg':>6} {'DP costo':>10} {'Base costo':>10} {'TT costo':>10} {'DP<Base%':>9} {'DP<SW%':>8}"
    print(header)
    print("─" * 88)

    for doc, res in zip(corpus, results):
        print(
            f"{doc['title'][:27]:<28} "
            f"{res['total_tokens']:>5} "
            f"{res['dp']['num_segments']:>6} "
            f"{res['dp']['total_cost']:>10,.0f} "
            f"{res['baseline']['total_cost']:>10,.0f} "
            f"{res['texttiling']['total_cost']:>10,.0f} "
            f"{res['reductions']['dp_vs_baseline_pct']:>8.2f}% "
            f"{res['reductions']['dp_vs_sliding_window_pct']:>7.2f}%"
        )

    print("─" * 88)

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

    stats = _compute_statistics(results)
    if stats:
        print(sep)
        print("Análisis estadístico — t-test pareado, Cohen's d, FDR (Benjamini-Hochberg):")
        print(f"  {'Comparación':<30} {'Reduc%':>7} {'IC95%':>20} {'t':>9} {'p':>10} {'p_FDR':>10} {'d':>7} {'Sig.':>5}")
        print("  " + "─" * 80)
        for c in stats["comparisons"]:
            sig = "✓" if c["significant"] else "✗"
            fdr_sig = "✓" if c["significant_fdr"] else "✗"
            print(
                f"  {c['comparison']:<30} "
                f"{c['mean_reduction_pct']:>6.2f}% "
                f"[{c['ci95_low']:>5.2f}%,{c['ci95_high']:>5.2f}%] "
                f"{c['t_stat']:>9.4f} "
                f"{c['p_value']:>10.4f} "
                f"{c['p_value_fdr']:>10.4f} "
                f"{c['cohens_d']:>7.4f} "
                f"{sig}/{fdr_sig}"
            )
        print("  Sig. = p<0.05 sin corr. / con FDR")

    print(sep)
    print("Métricas de frontera — Pk y WindowDiff (referencia = DP Óptimo, 0 = acuerdo perfecto):")
    print(f"  {'Texto':<28} {'Pk Base':>8} {'WD Base':>8} {'Pk SW':>7} {'WD SW':>7} {'Pk TT':>7} {'WD TT':>7}")
    print("  " + "─" * 70)
    for doc, res in zip(corpus, results):
        bm = res.get("boundary_metrics", {})
        def _fmt(v):
            return f"{v:.4f}" if v is not None else "   N/A"
        bm_base = bm.get("baseline", {})
        bm_sw   = bm.get("sliding_window", {})
        bm_tt   = bm.get("texttiling", {})
        print(
            f"  {doc['title'][:27]:<28} "
            f"{_fmt(bm_base.get('pk_vs_dp')):>8} "
            f"{_fmt(bm_base.get('wd_vs_dp')):>8} "
            f"{_fmt(bm_sw.get('pk_vs_dp')):>7} "
            f"{_fmt(bm_sw.get('wd_vs_dp')):>7} "
            f"{_fmt(bm_tt.get('pk_vs_dp')):>7} "
            f"{_fmt(bm_tt.get('wd_vs_dp')):>7}"
        )
    # Promedios
    def _avg_bm(method, key):
        vals = [r.get("boundary_metrics", {}).get(method, {}).get(key) for r in results]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")
    print("  " + "─" * 70)
    print(
        f"  {'PROMEDIO':<28} "
        f"{_avg_bm('baseline','pk_vs_dp'):>8.4f} "
        f"{_avg_bm('baseline','wd_vs_dp'):>8.4f} "
        f"{_avg_bm('sliding_window','pk_vs_dp'):>7.4f} "
        f"{_avg_bm('sliding_window','wd_vs_dp'):>7.4f} "
        f"{_avg_bm('texttiling','pk_vs_dp'):>7.4f} "
        f"{_avg_bm('texttiling','wd_vs_dp'):>7.4f}"
    )

    print(sep)
    print("Calibración k*:")
    for doc, res in zip(corpus, results):
        cal = res["calibration"]
        valid = "✓" if cal["valid"] else "✗"
        sub = cal.get("relative_suboptimality")
        sub_str = f"  sub={sub:.4f}" if sub is not None else "  sub=N/A"
        print(
            f"  {doc['id']:<22} k_dp={cal['k_dp']:>2}  k*={cal['k_analytical']:>2}  "
            f"dev={cal['deviation_pct']:.0%}{sub_str}  [{valid}]"
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
    bar_colors = ["#378ADD" if r >= 0 else "#d62728" for r in reductions]
    ax2.bar(xs, reductions, color=bar_colors, width=18, alpha=0.85)
    ax2.set_title("Reducción de costo DP vs Baseline (%)")
    ax2.set_xlabel("lmax (tokens máximos por segmento)")
    ax2.set_ylabel("Reducción (%)")
    # Eje Y: incluir zona negativa si existe, con margen cómodo
    r_min = min(reductions)
    r_max = max(reductions)
    ax2.set_ylim(min(r_min - 8, -5), r_max + 12)
    ax2.axhline(0, color="#333", linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")
    for x, r in zip(xs, reductions):
        va = "bottom" if r >= 0 else "top"
        offset = 1.5 if r >= 0 else -1.5
        ax2.text(x, r + offset, f"{r:.1f}%", ha="center", va=va, fontsize=9)

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


def analyze_failures(corpus: list[dict], results: list[dict], params: dict) -> dict:
    """Identifica textos con indicadores de problema en los resultados del benchmark."""
    sep = "═" * 64

    invalid_cal: list[dict] = []
    dp_no_improve: list[dict] = []
    k_far: list[dict] = []
    overlap_ineffective: list[dict] = []
    irregular_segs: list[dict] = []

    for doc, res in zip(corpus, results):
        title = doc["title"]
        cal = res["calibration"]
        dp = res["dp"]
        ov = res["dp_overlap"]

        if not cal["valid"]:
            invalid_cal.append({
                "title": title,
                "k_dp": cal["k_dp"],
                "k_analytical": cal["k_analytical"],
                "suboptimality": cal.get("relative_suboptimality"),
            })

        dp_full = res["dp"]["full_cost"]
        base_full = res["baseline"]["full_cost"]
        if base_full > 0 and (base_full - dp_full) / base_full * 100 <= 0:
            dp_no_improve.append({"title": title, "dp_full": dp_full, "base_full": base_full})

        if cal["deviation_pct"] > 0.15:
            k_far.append({
                "title": title,
                "k_dp": cal["k_dp"],
                "k_analytical": cal["k_analytical"],
                "deviation_pct": cal["deviation_pct"],
            })

        if ov.get("total_overlap_tokens", 0) == 0:
            overlap_ineffective.append({"title": title})

        avg_t = dp["avg_tokens"]
        std_t = dp["std_tokens"]
        if avg_t > 0 and std_t > 0.5 * avg_t:
            irregular_segs.append({
                "title": title,
                "avg_tokens": avg_t,
                "std_tokens": std_t,
                "ratio": round(std_t / avg_t, 3),
            })

    print(sep)
    print("ANÁLISIS DE FALLOS")
    print(sep)

    def _print_group(label: str, items: list[dict], formatter) -> None:
        print(f"{label} ({len(items)}):")
        if items:
            for item in items:
                print(f"  - {formatter(item)}")
        else:
            print("  ninguno")

    _print_group(
        "Calibración inválida",
        invalid_cal,
        lambda x: (
            f"{x['title']}: k_dp={x['k_dp']}, k*={x['k_analytical']}, "
            f"suboptimality={x['suboptimality'] * 100:.0f}%"
            if x["suboptimality"] is not None
            else f"{x['title']}: k_dp={x['k_dp']}, k*={x['k_analytical']}, suboptimality=N/A"
        ),
    )
    print()

    _print_group(
        "DP no mejora sobre baseline",
        dp_no_improve,
        lambda x: f"{x['title']}: dp_full={x['dp_full']:,.0f}, base_full={x['base_full']:,.0f}",
    )
    print()

    _print_group(
        "k* lejano del analítico (dev>15%)",
        k_far,
        lambda x: (
            f"{x['title']}: k_dp={x['k_dp']}, k*={x['k_analytical']}, "
            f"dev={x['deviation_pct']:.0%}"
        ),
    )
    print()

    _print_group(
        "Overlap inefectivo",
        overlap_ineffective,
        lambda x: x["title"],
    )
    print()

    _print_group(
        "Segmentos irregulares (std > 0.5*avg)",
        irregular_segs,
        lambda x: (
            f"{x['title']}: avg={x['avg_tokens']:.1f}, std={x['std_tokens']:.1f}, "
            f"ratio={x['ratio']}"
        ),
    )
    print()

    problem_titles = set(
        [x["title"] for x in invalid_cal]
        + [x["title"] for x in dp_no_improve]
        + [x["title"] for x in k_far]
        + [x["title"] for x in overlap_ineffective]
        + [x["title"] for x in irregular_segs]
    )
    total = len(corpus)
    print(f"Resumen: {len(problem_titles)}/{total} textos tienen al menos un indicador de problema.")
    print(sep)

    return {
        "invalid_calibration": invalid_cal,
        "dp_no_improve_over_baseline": dp_no_improve,
        "k_far_from_analytical": k_far,
        "overlap_ineffective": overlap_ineffective,
        "irregular_segments": irregular_segs,
        "texts_with_problems": len(problem_titles),
        "total_texts": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark de segmentación")
    parser.add_argument("--lmin", type=int, default=20)
    parser.add_argument("--lmax", type=int, default=150)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--lambda", dest="coherence_lambda", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.03)
    parser.add_argument("--fixed-cost", type=float, default=1000.0)
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

    params = {
        "lmin": args.lmin, "lmax": args.lmax, "overlap": args.overlap,
        "model": args.model, "coherence_lambda": args.coherence_lambda,
        "overlap_mu": args.mu, "fixed_cost": args.fixed_cost,
    }

    print()
    failure_analysis = analyze_failures(corpus, results, params)

    stats = _compute_statistics(results)

    if args.save_json:
        def _avg_bm(method, key):
            vals = [r.get("boundary_metrics", {}).get(method, {}).get(key) for r in results]
            vals = [v for v in vals if v is not None]
            return round(float(np.mean(vals)), 6) if vals else None

        out = {
            "params": params,
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
                "avg_pk_baseline_vs_dp":      _avg_bm("baseline", "pk_vs_dp"),
                "avg_wd_baseline_vs_dp":      _avg_bm("baseline", "wd_vs_dp"),
                "avg_pk_sw_vs_dp":            _avg_bm("sliding_window", "pk_vs_dp"),
                "avg_wd_sw_vs_dp":            _avg_bm("sliding_window", "wd_vs_dp"),
                "avg_pk_texttiling_vs_dp":    _avg_bm("texttiling", "pk_vs_dp"),
                "avg_wd_texttiling_vs_dp":    _avg_bm("texttiling", "wd_vs_dp"),
            },
            "statistics": stats,
            "failure_analysis": failure_analysis,
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
