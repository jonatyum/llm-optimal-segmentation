import numpy as np

from src.segmentation.embeddings import get_embeddings, cosine_similarity
from src.segmentation.splitter import split_sentences
from src.segmentation.tokenizer import count_tokens


def suggest_lambda(text: str) -> float:
    """Sugiere un valor de lambda basado en la variabilidad de coherencia del texto.

    Calcula el coeficiente de variación (CV = std/mean) de las similitudes coseno
    entre pares de oraciones consecutivas:
    - CV < 0.2  → coherencia muy uniforme  → lambda = 0.3
    - CV < 0.4  → coherencia moderada      → lambda = 0.5
    - CV >= 0.4 → coherencia muy variable  → lambda = 0.8
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.5

    embeddings = get_embeddings(sentences)
    scores = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    if mean_score == 0:
        return 0.5

    cv = std_score / mean_score

    if cv < 0.2:
        return 0.3
    elif cv < 0.4:
        return 0.5
    else:
        return 0.8


def estimate_optimal_k(
    text: str,
    model: str = "gpt-4o",
    fixed_cost: float = 1000.0,
    coherence_lambda: float = 0.0,
) -> int:
    """Estima el número óptimo de segmentos k*.

    Fórmula derivada minimizando:
        total_cost(k) = N²/k * (1 + λ*(1-coh_avg)) + k*Cf
    → k* = N * sqrt((1 + λ*(1-coh_avg)) / Cf)

    Para λ=0 equivale a la fórmula anterior k* = N/sqrt(Cf).
    """
    n_tokens = count_tokens(text, model=model)
    if fixed_cost <= 0:
        return 1

    scaling = 1.0
    if coherence_lambda > 0:
        sentences = split_sentences(text)
        if len(sentences) >= 2:
            embeddings = get_embeddings(sentences)
            sims = [
                cosine_similarity(embeddings[k], embeddings[k + 1])
                for k in range(len(embeddings) - 1)
            ]
            coh_avg = float(np.mean(sims))
            scaling = 1.0 + coherence_lambda * (1.0 - coh_avg)

    k_star = n_tokens * np.sqrt(scaling / fixed_cost)
    return max(1, round(k_star))


def calibrate_fixed_cost(llm_model: str = "gemma2:2b", num_trials: int = 5) -> float:
    """Calibra el costo fijo midiendo la latencia real del LLM local.

    Corre num_trials inferencias con un prompt mínimo y retorna
    mean(latencias_ms) * 0.1 como proxy del overhead fijo por segmento.
    """
    from src.segmentation.llm_client import run_inference

    latencies: list[float] = []
    for _ in range(num_trials):
        result = run_inference(prompt="Hi", model=llm_model)
        latencies.append(result.latency_ms)

    return float(np.mean(latencies)) * 0.1


def validate_calibration(
    text: str,
    lmin: int = 50,
    lmax: int = 200,
    coherence_lambda: float = 0.5,
    fixed_cost: float = 1000.0,
    model: str = "gpt-4o",
) -> dict:
    """Valida la calibración comparando el k* del DP 2D vs el k* analítico.

    Corre segment_dp_2d y compara con estimate_optimal_k.
    Un k* analítico es válido si el costo en dp2d.cost_by_k[k*] está dentro
    del 5% del costo óptimo — evita la comparación tautológica de solo verificar
    si k_analytical ≈ k_dp en número de segmentos.

    Retorna:
        k_dp                  : k óptimo encontrado por DP 2D
        k_analytical          : k* estimado analíticamente
        deviation_pct         : |k_dp - k_analytical| / max(k_analytical, 1)
        relative_suboptimality: (cost[k*] - cost[k_dp]) / cost[k_dp], o None si k* ∉ cost_by_k
        valid                 : relative_suboptimality <= 0.05
    """
    from src.segmentation.dp import segment_dp_2d

    dp2d_result = segment_dp_2d(
        text,
        lmin=lmin,
        lmax=lmax,
        model=model,
        coherence_lambda=coherence_lambda,
        fixed_cost=fixed_cost,
    )
    k_dp = dp2d_result.num_segments
    k_analytical = estimate_optimal_k(
        text, model=model, fixed_cost=fixed_cost, coherence_lambda=coherence_lambda
    )

    deviation_pct = abs(k_dp - k_analytical) / max(k_analytical, 1)

    if k_analytical in dp2d_result.cost_by_k:
        cost_at_k_star = dp2d_result.cost_by_k[k_analytical]
        cost_optimal = dp2d_result.cost_by_k[k_dp]
        relative_suboptimality = (cost_at_k_star - cost_optimal) / max(cost_optimal, 1e-9)
        valid = relative_suboptimality <= 0.05
    else:
        relative_suboptimality = float("inf")
        valid = False

    return {
        "k_dp": k_dp,
        "k_analytical": k_analytical,
        "deviation_pct": round(deviation_pct, 4),
        "relative_suboptimality": (
            round(relative_suboptimality, 4)
            if relative_suboptimality != float("inf")
            else None
        ),
        "valid": valid,
    }
