from dataclasses import dataclass
from src.segmentation.models import SegmentationResult
from src.segmentation.embeddings import segment_coherence, get_embeddings
from src.segmentation.dp import (
    _build_similarity_arrays,
    _avg_coherence,
    _build_cumulative_tokens,
    _compute_cost,
    DEFAULT_FIXED_COST,
    DEFAULT_LAMBDA,
)
from src.segmentation.tokenizer import count_tokens_batch


@dataclass
class SegmentationMetrics:
    num_segments: int
    total_cost: float
    avg_tokens_per_segment: float
    std_tokens_per_segment: float
    min_tokens: int
    max_tokens: int
    total_tokens: int
    avg_coherence: float


@dataclass
class ComparisonReport:
    dp: SegmentationMetrics
    baseline: SegmentationMetrics
    cost_reduction_pct: float
    segment_diff: int
    coherence_improvement: float


def compute_metrics(result: SegmentationResult) -> SegmentationMetrics:
    token_counts = [s.token_count for s in result.segments]
    n = len(token_counts)
    total = sum(token_counts)
    avg = total / n
    std = (sum((t - avg) ** 2 for t in token_counts) / n) ** 0.5

    coherence_scores = [
        segment_coherence(s.sentences)
        for s in result.segments
    ]
    avg_coherence = sum(coherence_scores) / len(coherence_scores)

    return SegmentationMetrics(
        num_segments=n,
        total_cost=result.total_cost,
        avg_tokens_per_segment=round(avg, 2),
        std_tokens_per_segment=round(std, 2),
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        total_tokens=total,
        avg_coherence=round(avg_coherence, 4),
    )


def compare(
    dp_result: SegmentationResult,
    baseline_result: SegmentationResult,
) -> ComparisonReport:
    dp_metrics = compute_metrics(dp_result)
    baseline_metrics = compute_metrics(baseline_result)

    cost_reduction = (
        (baseline_metrics.total_cost - dp_metrics.total_cost)
        / baseline_metrics.total_cost
        * 100
    )

    coherence_improvement = (
        dp_metrics.avg_coherence - baseline_metrics.avg_coherence
    )

    return ComparisonReport(
        dp=dp_metrics,
        baseline=baseline_metrics,
        cost_reduction_pct=round(cost_reduction, 2),
        segment_diff=baseline_metrics.num_segments - dp_metrics.num_segments,
        coherence_improvement=round(coherence_improvement, 4),
    )


def recompute_full_cost(
    result: SegmentationResult,
    coherence_lambda: float = DEFAULT_LAMBDA,
    fixed_cost: float = DEFAULT_FIXED_COST,
    model: str = "gpt-4o",
) -> float:
    """Re-evalúa el costo total de cualquier segmentación con la función completa del DP.

    Permite comparación justa entre DP y baseline porque ambos se evalúan con
    la misma función: t² * (1 + λ*(1-coh)) + Cf, en lugar del total_cost que
    el baseline calcula como pura suma de t².
    """
    # Reconstruir la lista completa de oraciones preservando el orden
    all_sentences: list[str] = []
    for seg in result.segments:
        all_sentences.extend(seg.sentences)

    if not all_sentences:
        return 0.0

    token_lens = count_tokens_batch(all_sentences, model=model)
    embeddings = get_embeddings(all_sentences)
    _, prefix_sims = _build_similarity_arrays(embeddings)
    cumtok = _build_cumulative_tokens(token_lens)

    # Construir mapa oración → índice global para ubicar los cortes
    total_cost = 0.0
    sentence_offset = 0
    for seg in result.segments:
        n_sents = len(seg.sentences)
        start = sentence_offset
        end = sentence_offset + n_sents
        span = cumtok[end] - cumtok[start]
        total_cost += _compute_cost(
            token_count=span,
            prefix_sims=prefix_sims,
            start=start,
            end=end,
            coherence_lambda=coherence_lambda,
            fixed_cost=fixed_cost,
        )
        sentence_offset = end

    return total_cost


def compare_full_cost(
    dp_result: SegmentationResult,
    other_result: SegmentationResult,
    coherence_lambda: float = DEFAULT_LAMBDA,
    fixed_cost: float = DEFAULT_FIXED_COST,
    model: str = "gpt-4o",
) -> dict:
    """Compara dos segmentaciones usando la función de costo completa del DP.

    Retorna un dict con los costos recomputados y la reducción porcentual.
    """
    dp_cost = recompute_full_cost(dp_result, coherence_lambda, fixed_cost, model)
    other_cost = recompute_full_cost(other_result, coherence_lambda, fixed_cost, model)
    reduction_pct = (
        (other_cost - dp_cost) / other_cost * 100
        if other_cost > 0 else 0.0
    )
    return {
        "dp_cost": dp_cost,
        "other_cost": other_cost,
        "cost_reduction_pct": round(reduction_pct, 2),
    }
