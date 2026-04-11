from dataclasses import dataclass
import numpy as np
from src.segmentation.llm_client import run_inference, run_segmented_inference, InferenceResult
from src.segmentation.embeddings import get_embeddings, cosine_similarity


@dataclass
class SegmentEvaluation:
    segment_index: int
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    latency_ms: float
    response_coherence: float


@dataclass
class EvaluationReport:
    method: str
    num_segments: int
    total_latency_ms: float
    avg_latency_ms: float
    total_prompt_tokens: int
    total_response_tokens: int
    avg_response_coherence: float
    segment_evaluations: list[SegmentEvaluation]


def _compute_response_coherence(
    prompt: str,
    response: str,
) -> float:
    embeddings = get_embeddings([prompt, response])
    return cosine_similarity(embeddings[0], embeddings[1])


def evaluate_segmentation(
    segments: list[str],
    method: str = "dp",
    model: str = "gemma2:2b",
    system_prompt: str = "You are a helpful assistant. Summarize the following text concisely.",
) -> EvaluationReport:
    results = run_segmented_inference(
        segments=segments,
        model=model,
        system_prompt=system_prompt,
    )

    evaluations = []
    for i, result in enumerate(results):
        coherence = _compute_response_coherence(
            prompt=result.prompt,
            response=result.response,
        )
        evaluations.append(SegmentEvaluation(
            segment_index=i,
            prompt_tokens=result.prompt_tokens,
            response_tokens=result.response_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            response_coherence=round(coherence, 4),
        ))

    total_latency = sum(e.latency_ms for e in evaluations)
    avg_latency = total_latency / len(evaluations)
    total_prompt_tokens = sum(e.prompt_tokens for e in evaluations)
    total_response_tokens = sum(e.response_tokens for e in evaluations)
    avg_coherence = float(np.mean([e.response_coherence for e in evaluations]))

    return EvaluationReport(
        method=method,
        num_segments=len(segments),
        total_latency_ms=round(total_latency, 2),
        avg_latency_ms=round(avg_latency, 2),
        total_prompt_tokens=total_prompt_tokens,
        total_response_tokens=total_response_tokens,
        avg_response_coherence=round(avg_coherence, 4),
        segment_evaluations=evaluations,
    )


def compare_evaluations(
    dp_report: EvaluationReport,
    baseline_report: EvaluationReport,
) -> dict:
    latency_reduction = (
        (baseline_report.total_latency_ms - dp_report.total_latency_ms)
        / baseline_report.total_latency_ms
        * 100
    )
    token_reduction = (
        (baseline_report.total_prompt_tokens - dp_report.total_prompt_tokens)
        / baseline_report.total_prompt_tokens
        * 100
    )
    coherence_improvement = (
        dp_report.avg_response_coherence - baseline_report.avg_response_coherence
    )

    return {
        "latency_reduction_pct": round(latency_reduction, 2),
        "token_reduction_pct": round(token_reduction, 2),
        "coherence_improvement": round(coherence_improvement, 4),
        "dp_total_latency_ms": dp_report.total_latency_ms,
        "baseline_total_latency_ms": baseline_report.total_latency_ms,
        "dp_avg_coherence": dp_report.avg_response_coherence,
        "baseline_avg_coherence": baseline_report.avg_response_coherence,
    }