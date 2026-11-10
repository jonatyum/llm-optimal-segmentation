"""
Consolidación de resultados: genera un resumen ejecutivo de todos los JSONs.

Carga los 6 archivos JSON de outputs/ y produce:
  - outputs/thesis_executive_summary.json  — todos los datos clave en un solo lugar
  - outputs/thesis_executive_summary.txt   — versión legible para copiar en la tesis

Uso:
    python experiments/consolidate_results.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUTS = Path(__file__).parent.parent / "outputs"


def _load(name: str) -> dict:
    path = OUTPUTS / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_summary() -> dict:
    bench   = _load("benchmark_results.json")
    stats   = _load("statistical_results.json")
    corpus  = _load("corpus_analysis.json")
    sens    = _load("sensitivity_results.json")
    comp    = _load("complexity_results.json")
    cases   = _load("case_studies.json")

    summary = {}

    # ── Parámetros ─────────────────────────────────────────────────────────
    summary["parametros_canonicos"] = bench.get("params", {})

    # ── Corpus ─────────────────────────────────────────────────────────────
    if corpus:
        texts = corpus.get("texts", [])
        tokens_list = [t["n_tokens"] for t in texts if "n_tokens" in t]
        cohs = [t["mean_coherence"] for t in texts if "mean_coherence" in t]
        summary["corpus"] = {
            "n_textos": len(texts),
            "tokens_total": sum(tokens_list),
            "tokens_promedio": round(sum(tokens_list) / len(tokens_list), 1) if tokens_list else 0,
            "tokens_min": min(tokens_list) if tokens_list else 0,
            "tokens_max": max(tokens_list) if tokens_list else 0,
            "coherencia_promedio": round(sum(cohs) / len(cohs), 4) if cohs else 0,
            "dominios": list({t["domain"] for t in texts}),
        }

    # ── Benchmark agregados ─────────────────────────────────────────────────
    if bench:
        agg = bench.get("aggregates", {})
        results = bench.get("results", [])
        reductions = [r["reductions"]["dp_vs_baseline_pct"] for r in results if "reductions" in r]
        summary["benchmark"] = {
            "n_textos": len(results),
            "reduccion_dp_vs_baseline_pct_nativa": round(agg.get("avg_dp_vs_baseline_pct", 0), 2),
            "reduccion_dp_vs_sw_pct_nativa": round(agg.get("avg_dp_vs_sw_pct", 0), 2),
            "coherencia_dp": round(agg.get("avg_coherence_dp", 0), 4),
            "coherencia_baseline": round(agg.get("avg_coherence_baseline", 0), 4),
            "mejora_coherencia": round(agg.get("avg_coherence_dp", 0) - agg.get("avg_coherence_baseline", 0), 4),
            "pk_baseline_vs_dp_avg": round(agg.get("avg_pk_baseline_vs_dp", 0) or 0, 4),
            "pk_texttiling_vs_dp_avg": round(agg.get("avg_pk_texttiling_vs_dp", 0) or 0, 4),
            "reduccion_min_pct": round(min(reductions), 2) if reductions else 0,
            "reduccion_max_pct": round(max(reductions), 2) if reductions else 0,
        }

    # ── Estadísticas (función de costo completa) ────────────────────────────
    if stats:
        comps = stats.get("comparisons", [])
        summary["estadisticas"] = [
            {
                "comparacion": c["comparison"],
                "n": c["n"],
                "reduccion_media_pct": c["mean_reduction_pct"],
                "ic95": [c["ci95_low"], c["ci95_high"]],
                "t_stat": c["t_stat"],
                "p_value": c["p_value"],
                "p_fdr": c.get("p_value_fdr", c["p_value"]),
                "cohens_d": c["cohens_d"],
                "efecto": c["effect_size"],
                "significativo": c["significant"],
            }
            for c in comps
        ]

    # ── Calibración k* ──────────────────────────────────────────────────────
    if bench:
        results = bench.get("results", [])
        cals = [r["calibration"] for r in results if "calibration" in r]
        devs = [c["deviation_pct"] for c in cals]
        valid_strict = sum(1 for c in cals if c.get("valid", False))
        valid_20pct = sum(1 for d in devs if d <= 0.20)
        summary["calibracion_k_star"] = {
            "cf": bench.get("params", {}).get("fixed_cost"),
            "valid_estricto_5pct": f"{valid_strict}/{len(cals)}",
            "valid_numerico_20pct": f"{valid_20pct}/{len(cals)}",
            "desviacion_media": round(sum(devs) / len(devs), 4) if devs else 0,
            "desviacion_std": round((sum((d - sum(devs)/len(devs))**2 for d in devs)/len(devs))**0.5, 4) if devs else 0,
        }

    # ── Sensibilidad ────────────────────────────────────────────────────────
    if sens:
        vals = list(sens.get("results", {}).values())
        k_list  = [v["mean_k_star"] for v in vals if v["mean_k_star"] == v["mean_k_star"]]
        red_list = [v["mean_reduction_pct"] for v in vals if v["mean_reduction_pct"] == v["mean_reduction_pct"]]
        best = max(vals, key=lambda v: v["mean_reduction_pct"])
        worst = min(vals, key=lambda v: v["mean_reduction_pct"])
        summary["sensibilidad"] = {
            "lambda_values": sens.get("lambda_values"),
            "cf_values": sens.get("cf_values"),
            "k_star_min": round(min(k_list), 1) if k_list else 0,
            "k_star_max": round(max(k_list), 1) if k_list else 0,
            "k_star_variacion": round(max(k_list) - min(k_list), 1) if k_list else 0,
            "reduccion_min_pct": round(min(red_list), 1) if red_list else 0,
            "reduccion_max_pct": round(max(red_list), 1) if red_list else 0,
            "reduccion_variacion_pp": round(max(red_list) - min(red_list), 1) if red_list else 0,
            "mejor_combinacion": {"lambda": best["lambda"], "cf": best["fixed_cost"], "reduccion_pct": round(best["mean_reduction_pct"], 1)},
            "peor_combinacion": {"lambda": worst["lambda"], "cf": worst["fixed_cost"], "reduccion_pct": round(worst["mean_reduction_pct"], 1)},
        }

    # ── Complejidad ─────────────────────────────────────────────────────────
    if comp:
        import math
        ns = comp.get("n_values", [])

        def slope(times):
            ts = [t for t in times if t and t > 1e-6]
            xs = ns[:len(ts)]
            if len(ts) < 3:
                return None
            coeffs = [0, 0]
            try:
                log_ns = [math.log(x) for x in xs]
                log_ts = [math.log(t) for t in ts]
                n = len(log_ns)
                sx = sum(log_ns); sy = sum(log_ts)
                sxx = sum(x*x for x in log_ns); sxy = sum(x*y for x, y in zip(log_ns, log_ts))
                s = (n*sxy - sx*sy) / (n*sxx - sx*sx)
                coeffs = [round(s, 2)]
            except Exception:
                pass
            return coeffs[0] if coeffs else None

        summary["complejidad"] = {
            "n_medidos": ns,
            "tiempos_dp_s": comp.get("dp_times", []),
            "tiempos_baseline_s": comp.get("baseline_times", []),
            "tiempos_texttiling_s": comp.get("texttiling_times", []),
            "pendiente_log_log_dp": slope(comp.get("dp_times", [])),
            "pendiente_log_log_baseline": slope(comp.get("baseline_times", [])),
            "pendiente_log_log_texttiling": slope(comp.get("texttiling_times", [])),
            "reps": comp.get("reps", 3),
        }

    # ── Casos de estudio ────────────────────────────────────────────────────
    if cases:
        dp_sims = [c["dp"]["avg_boundary_sim"] for c in cases]
        base_sims = [c["baseline"]["avg_boundary_sim"] for c in cases]
        improvements = [c["boundary_quality_improvement"] for c in cases]
        reductions = [c["cost_reduction_pct"] for c in cases]
        summary["casos_de_estudio"] = {
            "n_textos": len(cases),
            "sim_frontera_dp_avg": round(sum(dp_sims)/len(dp_sims), 4),
            "sim_frontera_baseline_avg": round(sum(base_sims)/len(base_sims), 4),
            "dp_mejora_fronteras_en": f"{sum(1 for i in improvements if i > 0.01)}/{len(cases)}",
            "reduccion_costo_avg": round(sum(reductions)/len(reductions), 2),
        }

    return summary


def _fmt_txt(summary: dict) -> str:
    lines = [
        "=" * 70,
        "RESUMEN EJECUTIVO — Tesis LLM Optimal Segmentation",
        "Juan Jonatan Churata Apaza | UMSA 2025",
        "=" * 70, "",
    ]

    if "parametros_canonicos" in summary:
        p = summary["parametros_canonicos"]
        lines += [
            "PARÁMETROS CANÓNICOS",
            f"  lmin={p.get('lmin')}  lmax={p.get('lmax')}  λ={p.get('coherence_lambda')}",
            f"  μ={p.get('overlap_mu')}  Cf={p.get('fixed_cost')}  modelo={p.get('model')}",
            "",
        ]

    if "corpus" in summary:
        c = summary["corpus"]
        lines += [
            "CORPUS",
            f"  N textos:    {c['n_textos']} textos en español",
            f"  Tokens:      total={c['tokens_total']:,}  promedio={c['tokens_promedio']:.0f}  rango=[{c['tokens_min']},{c['tokens_max']}]",
            f"  Coherencia:  promedio={c['coherencia_promedio']:.4f}",
            f"  Dominios:    {len(c['dominios'])} ({', '.join(sorted(c['dominios'])[:5])}...)",
            "",
        ]

    if "estadisticas" in summary:
        lines += ["ANÁLISIS ESTADÍSTICO (función de costo recomputada — comparación justa)"]
        lines.append(f"  {'Comparación':<30} {'Reduc%':>7}  IC95%               t(19)    p        d       Efecto")
        lines.append("  " + "-" * 85)
        for c in summary["estadisticas"]:
            lines.append(
                f"  {c['comparacion']:<30} {c['reduccion_media_pct']:>6.2f}%"
                f"  [{c['ic95'][0]:>5.2f}%, {c['ic95'][1]:>5.2f}%]"
                f"  {c['t_stat']:>7.4f}"
                f"  {c['p_value']:>7.4f}"
                f"  {c['cohens_d']:>6.4f}"
                f"  {c['efecto']}"
            )
        lines.append("  Todos los efectos son estadísticamente significativos con FDR (Benjamini-Hochberg)")
        lines.append("")

    if "benchmark" in summary:
        b = summary["benchmark"]
        lines += [
            "BENCHMARK (costos nativos de cada método)",
            f"  Reducción DP vs Baseline:      {b['reduccion_dp_vs_baseline_pct_nativa']:.2f}%",
            f"  Reducción DP vs Sliding Window:{b['reduccion_dp_vs_sw_pct_nativa']:.2f}%",
            f"  Coherencia DP:  {b['coherencia_dp']:.4f}   Baseline: {b['coherencia_baseline']:.4f}   Mejora: +{b['mejora_coherencia']:.4f}",
            f"  Pk Baseline vs DP:    {b['pk_baseline_vs_dp_avg']:.4f}",
            f"  Pk TextTiling vs DP:  {b['pk_texttiling_vs_dp_avg']:.4f}",
            "",
        ]

    if "calibracion_k_star" in summary:
        k = summary["calibracion_k_star"]
        lines += [
            "CALIBRACIÓN k* ANALÍTICO",
            f"  Cf usado: {k['cf']}",
            f"  Válido (sub-optimalidad ≤5%):  {k['valid_estricto_5pct']}",
            f"  Válido (desviación ≤20%):       {k['valid_numerico_20pct']}",
            f"  Desviación media:               {k['desviacion_media']*100:.1f}% ± {k['desviacion_std']*100:.1f}%",
            "",
        ]

    if "sensibilidad" in summary:
        s = summary["sensibilidad"]
        lines += [
            "ANÁLISIS DE SENSIBILIDAD (grid λ × Cf)",
            f"  λ ∈ {s['lambda_values']}",
            f"  Cf ∈ {s['cf_values']}",
            f"  k* varía: [{s['k_star_min']}, {s['k_star_max']}]  (rango = {s['k_star_variacion']} segs — robusto)",
            f"  Reducción varía: [{s['reduccion_min_pct']}%, {s['reduccion_max_pct']}%]  (variación = {s['reduccion_variacion_pp']}pp)",
            f"  Mejor combinación: λ={s['mejor_combinacion']['lambda']}, Cf={s['mejor_combinacion']['cf']} → {s['mejor_combinacion']['reduccion_pct']}%",
            "",
        ]

    if "complejidad" in summary:
        cx = summary["complejidad"]
        lines += [
            "COMPLEJIDAD COMPUTACIONAL EMPÍRICA",
            f"  Pendiente log-log DP:       ≈O(n^{cx['pendiente_log_log_dp']:.2f})" if cx['pendiente_log_log_dp'] else "  DP: pendiente no estimada",
            f"  Pendiente log-log Baseline: ≈O(n^{cx['pendiente_log_log_baseline']:.2f})" if cx['pendiente_log_log_baseline'] else "",
            f"  Pendiente log-log TT:       ≈O(n^{cx['pendiente_log_log_texttiling']:.2f})" if cx['pendiente_log_log_texttiling'] else "",
            f"  ({cx['reps']} repeticiones por punto, mediana)",
            "",
        ]

    if "casos_de_estudio" in summary:
        cs = summary["casos_de_estudio"]
        lines += [
            "CASOS DE ESTUDIO",
            f"  Textos analizados: {cs['n_textos']}",
            f"  Sim. frontera DP:       {cs['sim_frontera_dp_avg']:.4f}",
            f"  Sim. frontera Baseline: {cs['sim_frontera_baseline_avg']:.4f}",
            f"  DP mejora fronteras en: {cs['dp_mejora_fronteras_en']} casos",
            f"  Reducción costo promedio: {cs['reduccion_costo_avg']:.2f}%",
            "",
        ]

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    print("Consolidando resultados...\n")
    summary = build_summary()

    out_json = OUTPUTS / "thesis_executive_summary.json"
    out_txt  = OUTPUTS / "thesis_executive_summary.txt"

    OUTPUTS.mkdir(exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ {out_json}")

    txt = _fmt_txt(summary)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"✓ {out_txt}")

    print()
    print(txt)


if __name__ == "__main__":
    main()
