from dataclasses import dataclass
from src.segmentation.models import SegmentationResult
from src.segmentation.embeddings import segment_coherence


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