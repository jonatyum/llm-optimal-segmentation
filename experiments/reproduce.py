"""
Reproducibilidad: genera todos los resultados y figuras de la tesis.

Uso:
    python experiments/reproduce.py
    python experiments/reproduce.py --out outputs/results.json

Parámetros canónicos del experimento (los mismos de la tesis):
    lmin=20, lmax=150, lambda=0.5, mu=0.03, Cf=1000, model=gpt-4o
"""

import subprocess
import sys
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANONICAL = dict(
    lmin=20,
    lmax=150,
    overlap=30,
    model="gpt-4o",
    coherence_lambda=0.5,
    mu=0.03,
    fixed_cost=1000.0,
)

OUTPUT_JSON = os.path.join(ROOT, "outputs", "benchmark_results.json")
OUTPUT_LMAX = os.path.join(ROOT, "outputs", "cost_vs_lmax.png")
OUTPUT_TOKENS = os.path.join(ROOT, "outputs", "comparison_tokens.png")


def run():
    out_arg = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUTPUT_JSON

    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "run_benchmark.py"),
        "--lmin",    str(CANONICAL["lmin"]),
        "--lmax",    str(CANONICAL["lmax"]),
        "--overlap", str(CANONICAL["overlap"]),
        "--model",   CANONICAL["model"],
        "--lambda",  str(CANONICAL["coherence_lambda"]),
        "--mu",      str(CANONICAL["mu"]),
        "--fixed-cost", str(CANONICAL["fixed_cost"]),
        "--save-json", out_arg,
    ]

    print("=" * 60)
    print("Reproducibilidad — Parámetros canónicos")
    print("=" * 60)
    for k, v in CANONICAL.items():
        print(f"  {k:<20} {v}")
    print(f"\nJSON → {out_arg}")
    print(f"Figuras → {OUTPUT_LMAX}")
    print(f"         {OUTPUT_TOKENS}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

    if os.path.exists(out_arg):
        with open(out_arg, encoding="utf-8") as f:
            data = json.load(f)
        agg = data["aggregates"]
        print("\n" + "=" * 60)
        print("Resultados agregados (para la tesis)")
        print("=" * 60)
        print(f"  DP vs Baseline (reducción costo):   {agg['avg_dp_vs_baseline_pct']:.2f}%")
        print(f"  DP vs Sliding Window (reducción):   {agg['avg_dp_vs_sw_pct']:.2f}%")
        print(f"  Coherencia promedio DP:              {agg['avg_coherence_dp']:.4f}")
        print(f"  Coherencia promedio Baseline:        {agg['avg_coherence_baseline']:.4f}")
        print(f"  Mejora coherencia:                  +{agg['avg_coherence_dp'] - agg['avg_coherence_baseline']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    run()
