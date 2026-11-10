"""
Caracterización del corpus: estadísticas, desglose por dominio y análisis de coherencia.

Analiza los 20 textos de data/corpus.json y computa por texto:
  - Tokens totales, palabras, oraciones, tokens/oración promedio
  - Coherencia semántica: media, std y CV de similitudes coseno consecutivas
  - λ sugerido (heurística basada en CV de coherencia)
  - k* estimado analíticamente

Genera:
  - Tabla resumen por texto y por dominio
  - Gráficos: distribución de tokens, coherencia por dominio, k* vs N, λ sugerido

Uso:
    python experiments/corpus_analysis.py
    python experiments/corpus_analysis.py --no-plots
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
from collections import defaultdict

from src.segmentation.splitter import split_sentences
from src.segmentation.tokenizer import count_tokens
from src.segmentation.embeddings import get_embeddings, cosine_similarity, clear_embedding_cache
from src.segmentation.calibration import suggest_lambda, estimate_optimal_k

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus.json",
)
OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)


def analyze_text(doc: dict, model: str = "gpt-4o") -> dict:
    text = doc["text"]
    sentences = split_sentences(text)
    n_sentences = len(sentences)
    n_tokens = count_tokens(text, model=model)
    n_words = len(text.split())
    avg_sent_tokens = n_tokens / max(n_sentences, 1)

    # Coherencia: similitudes coseno entre pares consecutivos
    coherence_scores: list[float] = []
    if n_sentences >= 2:
        embeddings = get_embeddings(sentences)
        for i in range(len(embeddings) - 1):
            coherence_scores.append(cosine_similarity(embeddings[i], embeddings[i + 1]))

    mean_coh = float(np.mean(coherence_scores)) if coherence_scores else 1.0
    std_coh = float(np.std(coherence_scores)) if coherence_scores else 0.0
    cv_coh = std_coh / mean_coh if mean_coh > 0 else 0.0

    lam = suggest_lambda(text)
    k_star = estimate_optimal_k(text, model=model, fixed_cost=1000.0, coherence_lambda=lam)

    return {
        "id": doc["id"],
        "title": doc["title"],
        "domain": doc["domain"],
        "length_category": doc.get("length_category", "?"),
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "avg_sent_tokens": round(avg_sent_tokens, 2),
        "mean_coherence": round(mean_coh, 4),
        "std_coherence": round(std_coh, 4),
        "cv_coherence": round(cv_coh, 4),
        "suggested_lambda": lam,
        "estimated_k_star": k_star,
        "coherence_scores": coherence_scores,
    }


def aggregate_by_domain(stats: list[dict]) -> dict:
    by_domain: dict[str, list] = defaultdict(list)
    for s in stats:
        by_domain[s["domain"]].append(s)

    agg = {}
    for domain, items in sorted(by_domain.items()):
        def _m(key):
            vals = [x[key] for x in items]
            return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

        agg[domain] = {
            "n_texts": len(items),
            "n_tokens": _m("n_tokens"),
            "n_sentences": _m("n_sentences"),
            "avg_sent_tokens": _m("avg_sent_tokens"),
            "mean_coherence": _m("mean_coherence"),
            "cv_coherence": _m("cv_coherence"),
            "suggested_lambda": _m("suggested_lambda"),
            "estimated_k_star": _m("estimated_k_star"),
        }
    return agg


def print_summary(stats: list[dict], agg: dict) -> None:
    sep = "─" * 110
    print(sep)
    print(f"{'CARACTERIZACIÓN DEL CORPUS':^110}")
    print(sep)
    print(f"  {'ID':<24} {'Dom.':<18} {'Cat':>4} {'Tok':>5} {'Sent':>5} "
          f"{'T/S':>5} {'Coh.avg':>8} {'Coh.cv':>7} {'λ':>5} {'k*':>4}")
    print("  " + sep)

    for s in stats:
        print(
            f"  {s['id']:<24} {s['domain']:<18} {s['length_category']:>4} "
            f"{s['n_tokens']:>5} {s['n_sentences']:>5} "
            f"{s['avg_sent_tokens']:>5.1f} {s['mean_coherence']:>8.4f} "
            f"{s['cv_coherence']:>7.4f} {s['suggested_lambda']:>5.1f} "
            f"{s['estimated_k_star']:>4}"
        )

    print(sep)
    # Totales
    total_tok = sum(s["n_tokens"] for s in stats)
    avg_tok = np.mean([s["n_tokens"] for s in stats])
    avg_coh = np.mean([s["mean_coherence"] for s in stats])
    avg_cv = np.mean([s["cv_coherence"] for s in stats])
    print(f"  {'GLOBAL':<24} {'(20 textos)':<18} {'':>4} "
          f"{total_tok:>5} {'':>5} {'':>5} {avg_coh:>8.4f} {avg_cv:>7.4f}")
    print(f"  Tokens totales: {total_tok:,} | Promedio por texto: {avg_tok:.0f}")

    print("\n" + sep)
    print(f"{'AGREGADOS POR DOMINIO':^110}")
    print(sep)
    print(f"  {'Dominio':<20} {'N':>3} {'Tok avg±std':>16} {'Sent avg':>9} "
          f"{'Coh.avg':>8} {'Coh.cv':>8} {'λ avg':>7} {'k* avg':>7}")
    print("  " + sep)

    for domain, v in agg.items():
        tok_m, tok_s = v["n_tokens"]
        coh_m, _ = v["mean_coherence"]
        cv_m, _ = v["cv_coherence"]
        lam_m, _ = v["suggested_lambda"]
        k_m, _ = v["estimated_k_star"]
        sent_m, _ = v["n_sentences"]
        print(
            f"  {domain:<20} {v['n_texts']:>3} "
            f"{tok_m:>8.0f}±{tok_s:<6.0f} {sent_m:>9.1f} "
            f"{coh_m:>8.4f} {cv_m:>8.4f} {lam_m:>7.2f} {k_m:>7.1f}"
        )
    print(sep)

    # Insights
    all_cvs = [(s["domain"], s["cv_coherence"]) for s in stats]
    most_variable = max(all_cvs, key=lambda x: x[1])
    least_variable = min(all_cvs, key=lambda x: x[1])
    print(f"\nTexto más variable (CV alto → λ grande):  {most_variable[0]} (CV={most_variable[1]:.4f})")
    print(f"Texto más uniforme (CV bajo → λ pequeño): {least_variable[0]} (CV={least_variable[1]:.4f})")

    all_coh = [(s["id"], s["mean_coherence"]) for s in stats]
    most_coh = max(all_coh, key=lambda x: x[1])
    least_coh = min(all_coh, key=lambda x: x[1])
    print(f"Texto más coherente: {most_coh[0]} (coh={most_coh[1]:.4f})")
    print(f"Texto menos coherente: {least_coh[0]} (coh={least_coh[1]:.4f})")
    print(sep)


def _plot_corpus(stats: list[dict], agg: dict, out_dir: str) -> None:
    domains = list(agg.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    color_map = {d: c for d, c in zip(domains, colors)}
    # Etiquetas cortas para ejes X
    short_labels = [d.replace("Software Engineering", "Soft.Eng.")
                     .replace("Machine Learning", "ML")
                     .replace("Deep Learning", "Deep L.")
                     [:11] for d in domains]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    # 1. Tokens por dominio (boxplot)
    ax = axes[0, 0]
    domain_tokens = {d: [s["n_tokens"] for s in stats if s["domain"] == d] for d in domains}
    bp = ax.boxplot(
        [domain_tokens[d] for d in domains],
        tick_labels=short_labels,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title("Distribución de tokens por dominio", fontsize=11, pad=10)
    ax.set_ylabel("Tokens")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # 2. Coherencia media por dominio
    ax = axes[0, 1]
    means = [agg[d]["mean_coherence"][0] for d in domains]
    stds  = [agg[d]["mean_coherence"][1] for d in domains]
    ax.bar(range(len(domains)), means, yerr=stds, color=colors, alpha=0.8,
           capsize=4, error_kw=dict(linewidth=1.5))
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=8)
    ax.set_title("Coherencia semántica media por dominio\n(media ± std de similitudes coseno)", fontsize=10, pad=10)
    ax.set_ylabel("Coherencia promedio")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

    # 3. k* vs N (tokens totales) scatter — sin etiquetas por punto, solo leyenda de dominio
    ax = axes[1, 0]
    xs = np.array([s["n_tokens"] for s in stats], dtype=float)
    ys = np.array([s["estimated_k_star"] for s in stats], dtype=float)

    # Agrupar por dominio para una leyenda limpia
    for d, color in zip(domains, colors):
        pts = [(s["n_tokens"], s["estimated_k_star"]) for s in stats if s["domain"] == d]
        if pts:
            px, py = zip(*pts)
            ax.scatter(px, py, color=color, s=75, zorder=3, alpha=0.87, label=d)

    # Curva teórica k* = N/sqrt(Cf=2000)
    x_range = np.linspace(xs.min(), xs.max(), 100)
    k_theory = x_range / np.sqrt(2000.0)
    ax.plot(x_range, k_theory, "k--", linewidth=1.4, alpha=0.6, label="k*=N/√Cf (Cf=2000, λ=0)")

    ax.set_xlabel("N (tokens totales del texto)", fontsize=10)
    ax.set_ylabel("k* estimado (segmentos óptimos)", fontsize=10)
    ax.set_title("k* analítico vs longitud del texto", fontsize=11, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    # 4. λ sugerido por dominio (CV de coherencia)
    ax = axes[1, 1]
    cv_means  = [agg[d]["cv_coherence"][0] for d in domains]
    lam_means = [agg[d]["suggested_lambda"][0] for d in domains]
    x = np.arange(len(domains))
    w = 0.35
    b1 = ax.bar(x - w / 2, cv_means,  w, label="CV coherencia", color="#378ADD", alpha=0.82)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, lam_means, w, label="λ sugerido",   color="#EF9F27", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=8)
    ax.set_ylabel("CV de coherencia", fontsize=10)
    ax2.set_ylabel("λ sugerido", fontsize=10)
    ax2.set_ylim(0, max(lam_means) * 1.4)
    ax.set_title("CV de coherencia y λ sugerido por dominio", fontsize=10, pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    # Leyenda combinada manual
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#378ADD", alpha=0.82, label="CV coherencia"),
                 Patch(color="#EF9F27", alpha=0.82, label="λ sugerido")],
        fontsize=9, loc="upper right",
    )

    plt.suptitle(
        f"Caracterización del corpus — {len(stats)} textos en español",
        fontsize=14, y=1.01,
    )
    out_path = os.path.join(out_dir, "corpus_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Análisis del corpus")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Analizando {len(corpus)} textos...\n")
    stats = []
    for i, doc in enumerate(corpus, 1):
        print(f"  [{i:>2}/{len(corpus)}] {doc['id']}...", end=" ", flush=True)
        s = analyze_text(doc, model=args.model)
        stats.append(s)
        print(f"coh={s['mean_coherence']:.3f}  λ={s['suggested_lambda']}  k*={s['estimated_k_star']}")
        clear_embedding_cache()

    agg = aggregate_by_domain(stats)
    print()
    print_summary(stats, agg)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Serializar sin los scores completos (muy grandes)
    stats_out = [{k: v for k, v in s.items() if k != "coherence_scores"} for s in stats]
    out_json = os.path.join(OUTPUTS_DIR, "corpus_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"texts": stats_out, "by_domain": agg}, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados: {out_json}")

    if not args.no_plots:
        _plot_corpus(stats, agg, OUTPUTS_DIR)


if __name__ == "__main__":
    main()
