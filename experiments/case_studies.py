"""
Casos de estudio cualitativos: DP Óptimo vs Baseline, corte a corte.

Selecciona textos representativos del corpus y muestra lado a lado:
  - Dónde corta cada método (posición en oraciones)
  - Similitud coseno en cada frontera (menor = mejor corte semántico)
  - Previews del texto a ambos lados del corte
  - Estadísticas de calidad: avg coherencia en fronteras DP vs Baseline

Un "buen" corte tiene baja similitud coseno en la frontera (ruptura semántica clara).
Un "mal" corte tiene alta similitud (corta en medio de una idea continua).

Genera:
  - Reporte de texto en consola (apto para tesis)
  - outputs/case_studies.json con datos estructurados
  - outputs/case_studies.png con visualización de fronteras

Uso:
    python experiments/case_studies.py
    python experiments/case_studies.py --n-texts 5
    python experiments/case_studies.py --ids llm_inference nlp_segmentation algorithms_dp
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
import matplotlib.patches as mpatches

from src.segmentation.dp import segment_dp
from src.segmentation.baseline import segment_baseline
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity, clear_embedding_cache

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus.json",
)
OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)

# IDs representativos: diversidad de dominio y longitud
DEFAULT_IDS = [
    "llm_inference",
    "nlp_segmentation",
    "algorithms_dp",
    "history_roman_empire",
    "biology_cell_metabolism",
    "economics_market",
]


def _boundary_coherences(sentences: list[str], result) -> list[dict]:
    """Calcula la similitud coseno en cada frontera de segmentación."""
    if not sentences:
        return []
    embeddings = get_embeddings(sentences)
    boundaries = []
    offset = 0
    segs = result.segments
    for i, seg in enumerate(segs[:-1]):
        n = len(seg.sentences)
        last_idx = offset + n - 1
        first_idx_next = offset + n
        if first_idx_next < len(embeddings):
            sim = cosine_similarity(embeddings[last_idx], embeddings[first_idx_next])
            last_preview = seg.sentences[-1][:80].strip()
            first_preview = segs[i + 1].sentences[0][:80].strip()
            boundaries.append({
                "after_segment": i,
                "after_sentence_idx": last_idx,
                "similarity": round(float(sim), 4),
                "last_sentence": last_preview,
                "first_sentence": first_preview,
            })
        offset += n
    return boundaries


def analyze_case(
    doc: dict,
    lmin: int = 20,
    lmax: int = 150,
    coherence_lambda: float = 0.5,
    model: str = "gpt-4o",
) -> dict:
    text = doc["text"]
    sentences = split_sentences(text)

    dp_result = segment_dp(text, lmin=lmin, lmax=lmax, model=model,
                           coherence_lambda=coherence_lambda, fixed_cost=1000.0)
    base_result = segment_baseline(text, lmin=lmin, lmax=lmax, model=model)

    dp_bounds = _boundary_coherences(sentences, dp_result)
    base_bounds = _boundary_coherences(sentences, base_result)

    dp_avg_sim = float(np.mean([b["similarity"] for b in dp_bounds])) if dp_bounds else 1.0
    base_avg_sim = float(np.mean([b["similarity"] for b in base_bounds])) if base_bounds else 1.0

    cost_reduction = (
        (base_result.total_cost - dp_result.total_cost) / base_result.total_cost * 100
        if base_result.total_cost > 0 else 0.0
    )

    return {
        "id": doc["id"],
        "title": doc["title"],
        "domain": doc["domain"],
        "n_sentences": len(sentences),
        "dp": {
            "num_segments": dp_result.num_segments,
            "total_cost": round(dp_result.total_cost, 0),
            "avg_boundary_sim": round(dp_avg_sim, 4),
            "boundaries": dp_bounds,
        },
        "baseline": {
            "num_segments": base_result.num_segments,
            "total_cost": round(base_result.total_cost, 0),
            "avg_boundary_sim": round(base_avg_sim, 4),
            "boundaries": base_bounds,
        },
        "cost_reduction_pct": round(cost_reduction, 2),
        "boundary_quality_improvement": round(base_avg_sim - dp_avg_sim, 4),
    }


def print_case(case: dict, verbose: bool = True) -> None:
    sep = "═" * 88
    thin = "─" * 88
    dp = case["dp"]
    base = case["baseline"]

    print(sep)
    print(f"  CASO: {case['title']}")
    print(f"  Dominio: {case['domain']}  |  Oraciones: {case['n_sentences']}")
    print(sep)

    print(f"  {'':30} {'DP Óptimo':>18}  {'Baseline':>18}")
    print(f"  {'Segmentos':30} {dp['num_segments']:>18}  {base['num_segments']:>18}")
    print(f"  {'Costo total':30} {dp['total_cost']:>18,.0f}  {base['total_cost']:>18,.0f}")
    print(f"  {'Reducción de costo DP':30} {case['cost_reduction_pct']:>17.2f}%")
    print(f"  {'Sim. promedio en fronteras':30} {dp['avg_boundary_sim']:>18.4f}  {base['avg_boundary_sim']:>18.4f}")
    print(f"  {'Mejora calidad de corte':30} {case['boundary_quality_improvement']:>+17.4f}  (↓ mejor)")
    print(thin)

    quality_verdict = (
        "DP respeta mejor las fronteras semánticas"
        if case["boundary_quality_improvement"] > 0.01
        else "Calidad de corte similar entre métodos"
        if abs(case["boundary_quality_improvement"]) <= 0.01
        else "Baseline tiene fronteras ligeramente más claras"
    )
    print(f"  Veredicto: {quality_verdict}")
    print()

    if verbose and dp["boundaries"]:
        print(f"  Fronteras del DP Óptimo ({len(dp['boundaries'])} cortes):")
        print(f"  {'#':>3}  {'Sim.':>6}  {'Calidad':>9}  Último / Primero siguiente")
        print("  " + thin)
        for b in dp["boundaries"]:
            quality = "■ excelente" if b["similarity"] < 0.4 else "● bueno" if b["similarity"] < 0.65 else "○ regular"
            last = b["last_sentence"][:42].rstrip()
            nxt = b["first_sentence"][:42].rstrip()
            print(f"  {b['after_segment']:>3}  {b['similarity']:>6.4f}  {quality:>9}  ...{last!r}  →  {nxt!r}...")
        print()

    if verbose and base["boundaries"]:
        print(f"  Fronteras del Baseline ({len(base['boundaries'])} cortes):")
        print(f"  {'#':>3}  {'Sim.':>6}  {'Calidad':>9}  Último / Primero siguiente")
        print("  " + thin)
        for b in base["boundaries"]:
            quality = "■ excelente" if b["similarity"] < 0.4 else "● bueno" if b["similarity"] < 0.65 else "○ regular"
            last = b["last_sentence"][:42].rstrip()
            nxt = b["first_sentence"][:42].rstrip()
            print(f"  {b['after_segment']:>3}  {b['similarity']:>6.4f}  {quality:>9}  ...{last!r}  →  {nxt!r}...")
        print()


def print_global_summary(cases: list[dict]) -> None:
    sep = "═" * 88
    print(sep)
    print(f"{'RESUMEN GLOBAL — Calidad de Cortes':^88}")
    print(sep)
    print(f"  {'Texto':<30} {'Dom.':<16} {'DP segs':>7} {'Base segs':>9} "
          f"{'DP sim↓':>8} {'Base sim':>9} {'Mejora':>8} {'Reduc%':>8}")
    print("  " + "─" * 88)

    for c in cases:
        print(
            f"  {c['title'][:29]:<30} {c['domain'][:15]:<16} "
            f"{c['dp']['num_segments']:>7} {c['baseline']['num_segments']:>9} "
            f"{c['dp']['avg_boundary_sim']:>8.4f} {c['baseline']['avg_boundary_sim']:>9.4f} "
            f"{c['boundary_quality_improvement']:>+8.4f} {c['cost_reduction_pct']:>7.2f}%"
        )

    print("  " + "─" * 88)
    avg_improvement = np.mean([c["boundary_quality_improvement"] for c in cases])
    avg_reduction = np.mean([c["cost_reduction_pct"] for c in cases])
    better_cases = sum(1 for c in cases if c["boundary_quality_improvement"] > 0.01)
    print(f"  {'PROMEDIO':<46} "
          f"{np.mean([c['dp']['avg_boundary_sim'] for c in cases]):>8.4f} "
          f"{np.mean([c['baseline']['avg_boundary_sim'] for c in cases]):>9.4f} "
          f"{avg_improvement:>+8.4f} {avg_reduction:>7.2f}%")
    print()
    print(f"  DP mejora calidad de fronteras en {better_cases}/{len(cases)} casos "
          f"(sim promedio en fronteras ↓{abs(avg_improvement):.4f})")
    print(f"  Reducción de costo promedio: {avg_reduction:.2f}%")
    print(f"  Nota: Sim. de frontera ↓ = mejor (mayor ruptura semántica = corte más natural)")
    print(sep)


def _plot_cases(cases: list[dict], out_path: str) -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.5 * n))
    fig.subplots_adjust(hspace=0.55)
    if n == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        dp_sims   = [b["similarity"] for b in case["dp"]["boundaries"]]
        base_sims = [b["similarity"] for b in case["baseline"]["boundaries"]]

        x_dp   = np.arange(len(dp_sims))
        x_base = np.arange(len(base_sims))

        ax.scatter(x_dp,   dp_sims,   color="#378ADD", s=70, zorder=4, label="DP Óptimo", marker="o")
        ax.plot(   x_dp,   dp_sims,   color="#378ADD", linewidth=1.5, alpha=0.7)
        ax.scatter(x_base, base_sims, color="#EF9F27", s=55, zorder=4, label="Baseline",  marker="s")
        ax.plot(   x_base, base_sims, color="#EF9F27", linewidth=1.5, alpha=0.7, linestyle="--")

        # Líneas de medias
        if dp_sims:
            ax.axhline(np.mean(dp_sims),   color="#378ADD", linestyle=":", linewidth=1.2, alpha=0.55)
        if base_sims:
            ax.axhline(np.mean(base_sims), color="#EF9F27", linestyle=":", linewidth=1.2, alpha=0.55)

        # Zonas de referencia semántica (muy transparentes para no tapar datos)
        x_end = max(len(dp_sims), len(base_sims)) - 1
        ax.fill_between([0, x_end], 0,    0.40, alpha=0.04, color="green")
        ax.fill_between([0, x_end], 0.65, 1.0,  alpha=0.04, color="red")
        # Etiquetas de zona en el margen derecho
        ax.text(x_end + 0.15, 0.20, "bueno\n(<0.4)", fontsize=7, color="green",
                ha="left", va="center", alpha=0.7)
        ax.text(x_end + 0.15, 0.82, "regular\n(>0.65)", fontsize=7, color="red",
                ha="left", va="center", alpha=0.7)

        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, x_end + 1.5)
        ax.set_ylabel("Similitud coseno\nen frontera", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Índice de frontera (corte entre segmentos consecutivos)", fontsize=9)

        # Título en dos líneas: nombre del texto + estadísticas
        title_text = (f"{case['title'][:60]}  [{case['domain']}]")
        stats_text = (
            f"DP: {case['dp']['num_segments']} segs, sim_avg={case['dp']['avg_boundary_sim']:.3f}  |  "
            f"Baseline: {case['baseline']['num_segments']} segs, sim_avg={case['baseline']['avg_boundary_sim']:.3f}  |  "
            f"Reduc. costo: {case['cost_reduction_pct']:.1f}%"
        )
        ax.set_title(f"{title_text}\n{stats_text}", fontsize=9, loc="left", pad=8)
        ax.legend(fontsize=8.5, loc="upper right")

    plt.suptitle(
        "Calidad de cortes: similitud coseno en cada frontera\n"
        "Valores más bajos = ruptura semántica más clara = mejor corte",
        fontsize=12, y=1.005,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Casos de estudio cualitativos DP vs Baseline")
    parser.add_argument("--n-texts", type=int, default=6,
                        help="Número de textos a analizar (si no se especifican IDs)")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="IDs específicos del corpus a analizar")
    parser.add_argument("--lmin", type=int, default=20)
    parser.add_argument("--lmax", type=int, default=150)
    parser.add_argument("--lambda", dest="coherence_lambda", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--brief", action="store_true",
                        help="Omitir detalle de fronteras individuales")
    args = parser.parse_args()

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    if args.ids:
        selected = [d for d in corpus if d["id"] in args.ids]
        if not selected:
            print(f"ERROR: ninguno de los IDs {args.ids} encontrado en el corpus.")
            sys.exit(1)
    else:
        target_ids = DEFAULT_IDS[: args.n_texts]
        selected = [d for d in corpus if d["id"] in target_ids]
        if len(selected) < args.n_texts:
            # Completar con textos del corpus no seleccionados
            extras = [d for d in corpus if d["id"] not in target_ids]
            selected += extras[: args.n_texts - len(selected)]
        selected = selected[: args.n_texts]

    print(f"Analizando {len(selected)} textos (lmin={args.lmin}, lmax={args.lmax}, λ={args.coherence_lambda})\n")

    cases = []
    for i, doc in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {doc['id']}...", flush=True)
        case = analyze_case(doc, args.lmin, args.lmax, args.coherence_lambda, args.model)
        cases.append(case)
        clear_embedding_cache()

    print()
    for case in cases:
        print_case(case, verbose=not args.brief)
    print_global_summary(cases)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUTS_DIR, "case_studies.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados: {out_json}")

    if not args.no_plots:
        _plot_cases(cases, os.path.join(OUTPUTS_DIR, "case_studies.png"))


if __name__ == "__main__":
    main()
