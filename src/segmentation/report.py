from dataclasses import dataclass
from src.segmentation.dp import segment_dp, segment_dp_2d, DP2DResult
from src.segmentation.baseline import segment_baseline
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.metrics import compute_metrics, compare
from src.segmentation.evaluator import evaluate_segmentation


@dataclass
class FullReport:
    text_length_tokens: int
    num_sentences: int
    suggested_lambda: float
    dp_metrics: dict
    baseline_metrics: dict
    overlap_metrics: dict
    sliding_window_metrics: dict
    dp_2d_metrics: dict
    dp_2d_analytical: dict
    dp_vs_baseline: dict
    dp_vs_sliding_window: dict


def generate_full_report(
    text: str,
    lmin: int = 50,
    lmax: int = 200,
    overlap: int = 50,
    model: str = "gpt-4o",
    coherence_lambda: float = 0.5,
    overlap_mu: float = 0.3,
    fixed_cost: float = 1000.0,
    run_llm_evaluation: bool = False,
    llm_model: str = "gemma2:2b",
) -> FullReport:
    from src.segmentation.splitter import split_sentences
    from src.segmentation.tokenizer import count_tokens
    from src.segmentation.calibration import estimate_optimal_k, validate_calibration, suggest_lambda

    sentences = split_sentences(text)
    total_tokens = count_tokens(text, model=model)
    lambda_suggested = suggest_lambda(text)

    # correr todos los métodos
    dp_result = segment_dp(
        text, lmin=lmin, lmax=lmax, model=model,
        coherence_lambda=coherence_lambda,
        fixed_cost=fixed_cost,
    )
    baseline_result = segment_baseline(text, lmax=lmax, model=model)
    overlap_result = segment_dp_overlap(
        text, lmin=lmin, lmax=lmax, model=model,
        coherence_lambda=coherence_lambda,
        overlap_mu=overlap_mu,
        fixed_cost=fixed_cost,
    )
    sw_result = segment_sliding_window(text, lmax=lmax, overlap=overlap, model=model)
    dp2d_result = segment_dp_2d(
        text, lmin=lmin, lmax=lmax, model=model,
        coherence_lambda=coherence_lambda,
        fixed_cost=fixed_cost,
    )

    # calibración analítica
    k_analytical = estimate_optimal_k(text, model=model, fixed_cost=fixed_cost)
    calibration = validate_calibration(
        text, lmin=lmin, lmax=lmax,
        coherence_lambda=coherence_lambda,
        fixed_cost=fixed_cost,
        model=model,
    )

    # métricas estáticas
    dp_metrics = compute_metrics(dp_result)
    baseline_metrics = compute_metrics(baseline_result)

    dp_vs_baseline = compare(dp_result, baseline_result)
    dp_vs_sw = {
        "cost_reduction_pct": round(
            (sw_result.total_cost - dp_result.total_cost)
            / sw_result.total_cost * 100, 2
        ),
        "segment_diff": sw_result.num_segments - dp_result.num_segments,
        "overlap_token_diff": sw_result.total_overlap_tokens,
    }

    # métricas con LLM real (opcional)
    llm_dp = {}
    llm_baseline = {}
    llm_sw = {}

    if run_llm_evaluation:
        dp_segments = [s.text for s in dp_result.segments]
        base_segments = [s.text for s in baseline_result.segments]
        sw_segments = [s.text for s in sw_result.segments]

        llm_dp = evaluate_segmentation(dp_segments, method="dp", model=llm_model)
        llm_baseline = evaluate_segmentation(base_segments, method="baseline", model=llm_model)
        llm_sw = evaluate_segmentation(sw_segments, method="sliding_window", model=llm_model)

    return FullReport(
        text_length_tokens=total_tokens,
        num_sentences=len(sentences),
        suggested_lambda=lambda_suggested,
        dp_metrics={
            "num_segments": dp_metrics.num_segments,
            "total_cost": dp_metrics.total_cost,
            "avg_tokens": dp_metrics.avg_tokens_per_segment,
            "std_tokens": dp_metrics.std_tokens_per_segment,
            "avg_coherence": dp_metrics.avg_coherence,
            "llm_evaluation": llm_dp,
        },
        baseline_metrics={
            "num_segments": baseline_metrics.num_segments,
            "total_cost": baseline_metrics.total_cost,
            "avg_tokens": baseline_metrics.avg_tokens_per_segment,
            "std_tokens": baseline_metrics.std_tokens_per_segment,
            "avg_coherence": baseline_metrics.avg_coherence,
            "llm_evaluation": llm_baseline,
        },
        overlap_metrics={
            "num_segments": overlap_result.num_segments,
            "total_cost": overlap_result.total_cost,
            "total_overlap_tokens": overlap_result.total_overlap_tokens,
            "llm_evaluation": {},
        },
        sliding_window_metrics={
            "num_segments": sw_result.num_segments,
            "total_cost": sw_result.total_cost,
            "total_overlap_tokens": sw_result.total_overlap_tokens,
            "llm_evaluation": llm_sw,
        },
        dp_2d_metrics={
            "optimal_k": dp2d_result.num_segments,
            "total_cost": dp2d_result.total_cost,
            "cost_by_k": dp2d_result.cost_by_k,
            "k_analytical": k_analytical,
            "deviation_pct": calibration["deviation_pct"],
            "calibration_valid": calibration["valid"],
            "segments": [
                {
                    "index": s.index,
                    "token_count": s.token_count,
                    "num_sentences": len(s.sentences),
                }
                for s in dp2d_result.segments
            ],
        },
        dp_2d_analytical=calibration,
        dp_vs_baseline={
            "cost_reduction_pct": dp_vs_baseline.cost_reduction_pct,
            "coherence_improvement": dp_vs_baseline.coherence_improvement,
            "segment_diff": dp_vs_baseline.segment_diff,
        },
        dp_vs_sliding_window=dp_vs_sw,
    )
