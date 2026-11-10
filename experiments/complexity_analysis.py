"""
Análisis de complejidad computacional: tiempo de ejecución vs n (oraciones).

Genera textos sintéticos de n oraciones y mide el tiempo de ejecución de:
  - DP Óptimo (teórico: O(n * w), donde w = lmax/token_avg ≈ constante)
  - Baseline greedy (teórico: O(n))
  - TextTiling (teórico: O(n²) por cálculo de similitudes de bloque)

Cada medición se repite 3 veces; se reporta la mediana para robustez.
El embedding inicial se calienta antes de medir para no incluir latencia de carga.

Uso:
    python experiments/complexity_analysis.py
    python experiments/complexity_analysis.py --reps 5 --max-n 300
"""

import sys
import os
import time
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

from src.segmentation.dp import segment_dp
from src.segmentation.baseline import segment_baseline
from src.segmentation.texttiling import segment_texttiling
from src.segmentation.embeddings import clear_embedding_cache

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)

# Oraciones de ~8-12 tokens para textos bien formados
_BASE_SENTENCES = [
    "Los modelos de lenguaje procesan texto de forma secuencial usando transformadores.",
    "La segmentación óptima minimiza el costo total de inferencia del modelo.",
    "La programación dinámica garantiza encontrar la partición de costo mínimo.",
    "El costo de autoatención crece de forma cuadrática con la longitud del contexto.",
    "Los embeddings capturan la similitud semántica entre oraciones consecutivas.",
    "La ventana de contexto limita la cantidad de texto que el modelo puede procesar.",
    "El algoritmo divide el texto en fragmentos semánticamente cohesivos y manejables.",
    "La función de costo integra tokens al cuadrado y una penalización por incoherencia.",
    "Los transformadores capturan dependencias de largo alcance mediante autoatención.",
    "La calibración estima el número óptimo de segmentos dada la longitud del texto.",
]


def make_text(n_sentences: int) -> str:
    """Genera un texto con exactamente n oraciones bien formadas."""
    return " ".join(_BASE_SENTENCES[i % len(_BASE_SENTENCES)] for i in range(n_sentences))


def warmup() -> None:
    """Carga spaCy y sentence-transformers una vez para no contaminar las mediciones."""
    text = make_text(5)
    segment_dp(text, lmin=5, lmax=50, max_sentences=50)
    clear_embedding_cache()
    print("  Modelos cargados (warmup completado)")


def measure_method(fn, text: str, reps: int) -> float:
    """Ejecuta fn(text) reps veces y retorna la mediana del tiempo (segundos)."""
    times = []
    for _ in range(reps):
        clear_embedding_cache()
        t0 = time.perf_counter()
        fn(text)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_complexity(
    n_values: list[int],
    lmin: int,
    lmax: int,
    model: str,
    reps: int,
) -> dict:
    dp_times, base_times, tt_times = [], [], []

    for n in n_values:
        text = make_text(n)
        print(f"  n={n:>4} oraciones ... ", end="", flush=True)

        fn_dp = lambda t: segment_dp(t, lmin=lmin, lmax=lmax, model=model, max_sentences=n + 10)
        fn_base = lambda t: segment_baseline(t, lmin=lmin, lmax=lmax, model=model, max_sentences=n + 10)
        fn_tt = lambda t: segment_texttiling(t, lmin=lmin, lmax=lmax, model=model, max_sentences=n + 10)

        t_dp = measure_method(fn_dp, text, reps)
        t_base = measure_method(fn_base, text, reps)
        t_tt = measure_method(fn_tt, text, reps)

        dp_times.append(t_dp)
        base_times.append(t_base)
        tt_times.append(t_tt)

        print(f"DP={t_dp:.3f}s  Base={t_base:.3f}s  TT={t_tt:.3f}s")

    return {
        "n_values": n_values,
        "dp_times": dp_times,
        "baseline_times": base_times,
        "texttiling_times": tt_times,
        "reps": reps,
        "params": {"lmin": lmin, "lmax": lmax, "model": model},
    }


def _fit_polynomial(n_values: list[int], times: list[float], degree: int) -> np.ndarray:
    """Ajusta una curva polinomial y retorna valores predichos."""
    xs = np.array(n_values, dtype=float)
    ys = np.array(times, dtype=float)
    coeffs = np.polyfit(xs, ys, degree)
    return np.polyval(coeffs, xs)


def print_results(results: dict) -> None:
    sep = "─" * 68
    print(sep)
    print(f"{'ANÁLISIS DE COMPLEJIDAD — Tiempo vs n (oraciones)':^68}")
    print(sep)
    print(f"  {'n':>5}  {'DP (s)':>9}  {'Baseline (s)':>13}  {'TextTiling (s)':>14}  {'DP/Base':>7}")
    print("  " + sep)
    for n, dp, base, tt in zip(
        results["n_values"], results["dp_times"],
        results["baseline_times"], results["texttiling_times"]
    ):
        ratio = dp / base if base > 0 else float("nan")
        print(f"  {n:>5}  {dp:>9.4f}  {base:>13.4f}  {tt:>14.4f}  {ratio:>7.2f}×")
    print(sep)

    # Estimar pendiente log-log
    ns = np.array(results["n_values"], dtype=float)
    for label, times in [
        ("DP", results["dp_times"]),
        ("Baseline", results["baseline_times"]),
        ("TextTiling", results["texttiling_times"]),
    ]:
        ys = np.array(times, dtype=float)
        valid = ys > 1e-6
        if valid.sum() >= 3:
            slope, _ = np.polyfit(np.log(ns[valid]), np.log(ys[valid]), 1)
            print(f"  {label}: pendiente log-log ≈ {slope:.2f}  (O(n^{slope:.2f}))")
    print(sep)


def _plot_complexity(results: dict, out_path: str) -> None:
    ns = np.array(results["n_values"], dtype=float)
    dp_t = np.array(results["dp_times"])
    base_t = np.array(results["baseline_times"])
    tt_t = np.array(results["texttiling_times"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # ── Gráfico lineal ──────────────────────────────────────────────────────
    ax1.plot(ns, dp_t,   "o-",  color="#378ADD", linewidth=2.2, label="DP Óptimo",      markersize=6)
    ax1.plot(ns, base_t, "s--", color="#EF9F27", linewidth=2.0, label="Baseline greedy", markersize=6)
    ax1.plot(ns, tt_t,   "^:",  color="#7B5EA7", linewidth=2.0, label="TextTiling",      markersize=6)

    # Curvas teóricas escaladas al punto central
    n_mid    = ns[len(ns) // 2]
    dp_mid   = dp_t[len(dp_t) // 2]   if dp_t[len(dp_t) // 2]   > 0 else 1e-3
    base_mid = base_t[len(base_t) // 2] if base_t[len(base_t) // 2] > 0 else 1e-3

    theory_linear = base_mid * (ns / n_mid)
    theory_quad   = dp_mid   * (ns / n_mid) ** 2

    ax1.plot(ns, theory_linear, ":", color="#EF9F27", linewidth=1.2, alpha=0.5, label="O(n) teórico")
    ax1.plot(ns, theory_quad,   ":", color="#378ADD", linewidth=1.2, alpha=0.5, label="O(n²) teórico")

    ax1.set_xlabel("n (número de oraciones)", fontsize=11)
    ax1.set_ylabel("Tiempo de ejecución (segundos)", fontsize=11)
    ax1.set_title("Tiempo de ejecución vs número de oraciones", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ── Log-log ─────────────────────────────────────────────────────────────
    slopes = {}
    for times, color, label, marker in [
        (dp_t,   "#378ADD", "DP Óptimo",      "o"),
        (base_t, "#EF9F27", "Baseline greedy", "s"),
        (tt_t,   "#7B5EA7", "TextTiling",      "^"),
    ]:
        valid = times > 1e-6
        if valid.sum() >= 3:
            slope, _ = np.polyfit(np.log(ns[valid]), np.log(times[valid]), 1)
            slopes[label] = slope
            legend_label = f"{label}  [≈O(n^{slope:.2f})]"
        else:
            legend_label = label
        ax2.loglog(ns[valid], times[valid], f"{marker}-", color=color,
                   linewidth=2.2, label=legend_label, markersize=6)

    # Líneas guía
    ref_x = np.array([ns[0], ns[-1]])
    ref_n = float(ns[0])
    ref_dp   = float(dp_t[0])   if dp_t[0]   > 1e-6 else 1e-3
    ref_base = float(base_t[0]) if base_t[0] > 1e-6 else 1e-3
    ax2.loglog(ref_x, ref_base * (ref_x / ref_n),      ":", color="#EF9F27", linewidth=1, alpha=0.4)
    ax2.loglog(ref_x, ref_dp   * (ref_x / ref_n) ** 2, ":", color="#378ADD", linewidth=1, alpha=0.4)

    # Las pendientes ya están en la leyenda — no se necesitan anotaciones extra

    ax2.set_xlabel("n (escala log)", fontsize=11)
    ax2.set_ylabel("Tiempo (s, escala log)", fontsize=11)
    ax2.set_title("Escala log-log — pendiente = exponente de complejidad", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"Análisis de complejidad computacional · {results['reps']} repeticiones por punto · mediana",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Análisis de complejidad computacional")
    parser.add_argument("--lmin", type=int, default=10)
    parser.add_argument("--lmax", type=int, default=60)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--reps", type=int, default=3,
                        help="Repeticiones por punto de medición")
    parser.add_argument("--max-n", type=int, default=200,
                        help="Máximo número de oraciones a medir")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    n_values = [n for n in [10, 20, 30, 50, 75, 100, 150, 200] if n <= args.max_n]
    if not n_values:
        n_values = [10, 20]

    print(f"n a medir: {n_values}")
    print(f"Repeticiones por punto: {args.reps}")
    print(f"Parámetros DP: lmin={args.lmin}, lmax={args.lmax}\n")
    print("Calentando modelos...")
    warmup()
    print()

    results = run_complexity(n_values, args.lmin, args.lmax, args.model, args.reps)
    print()
    print_results(results)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUTS_DIR, "complexity_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados: {out_json}")

    if not args.no_plots:
        _plot_complexity(results, os.path.join(OUTPUTS_DIR, "complexity_analysis.png"))


if __name__ == "__main__":
    main()
