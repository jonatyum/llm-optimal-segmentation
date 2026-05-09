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


def estimate_optimal_k(text: str, model: str = "gpt-4o", fixed_cost: float = 1000.0) -> int:
    """Estima el número óptimo de segmentos k* usando la fórmula analítica.

    k* = round(n_tokens / sqrt(fixed_cost))

    Derivado de minimizar el costo cuadrático total:
    k * (n/k)^2 + k * fixed_cost → mínimo en k = n / sqrt(fixed_cost)
    """
    n_tokens = count_tokens(text, model=model)
    if fixed_cost <= 0:
        return 1
    k_star = round(n_tokens / np.sqrt(fixed_cost))
    return max(1, k_star)


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
    Retorna:
        k_dp          : k óptimo encontrado por DP 2D
        k_analytical  : k* estimado analíticamente
        deviation_pct : |k_dp - k_analytical| / k_analytical
        valid         : deviation_pct <= 0.20
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
    k_analytical = estimate_optimal_k(text, model=model, fixed_cost=fixed_cost)

    deviation_pct = abs(k_dp - k_analytical) / max(k_analytical, 1)

    return {
        "k_dp": k_dp,
        "k_analytical": k_analytical,
        "deviation_pct": round(deviation_pct, 4),
        "valid": deviation_pct <= 0.20,
    }
