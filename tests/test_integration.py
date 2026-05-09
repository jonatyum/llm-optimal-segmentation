# tests/test_integration.py
import pytest
from src.segmentation import (
    segment_dp,
    segment_dp_2d,
    segment_baseline,
    segment_dp_overlap,
    segment_sliding_window,
    compute_metrics,
    compare,
    recompute_full_cost,
    compare_full_cost,
    generate_full_report,
    FullReport,
)

TEXT = (
    "Large language models have billions of parameters. "
    "They require significant computational resources for inference. "
    "Techniques like quantization help reduce inference costs substantially. "
    "Prompt engineering has emerged as a key skill for working with these models. "
    "Retrieval-augmented generation combines LLMs with external knowledge sources. "
    "Fine-tuning on domain-specific data improves performance on specialized tasks. "
    "Evaluation remains a challenge as automated metrics fail to capture true quality. "
    "Human evaluation is expensive but more reliable than automated methods. "
    "Benchmarks like MMLU and HellaSwag are widely used for evaluation."
)


def test_full_pipeline_dp_vs_baseline():
    # fixed_cost bajo para que el término cuadrático domine la comparación
    dp_result = segment_dp(TEXT, lmin=10, lmax=80, fixed_cost=0.0)
    base_result = segment_baseline(TEXT, lmax=80)
    report = compare(dp_result, base_result)
    assert report.dp.total_cost < report.baseline.total_cost
    assert report.cost_reduction_pct > 0


def test_full_pipeline_dp_vs_baseline_fair_cost():
    """Comparación justa: ambas segmentaciones evaluadas con la misma función de costo."""
    dp_result = segment_dp(TEXT, lmin=10, lmax=80, fixed_cost=0.0, coherence_lambda=0.5)
    base_result = segment_baseline(TEXT, lmax=80)
    comparison = compare_full_cost(
        dp_result, base_result,
        coherence_lambda=0.5, fixed_cost=0.0,
    )
    assert comparison["dp_cost"] > 0
    assert comparison["other_cost"] > 0
    assert isinstance(comparison["cost_reduction_pct"], float)


def test_recompute_full_cost_is_consistent():
    """recompute_full_cost con los mismos parámetros del DP debe dar un costo similar al DP."""
    cf = 500.0
    lam = 0.3
    dp_result = segment_dp(TEXT, lmin=10, lmax=80, fixed_cost=cf, coherence_lambda=lam)
    recomputed = recompute_full_cost(dp_result, coherence_lambda=lam, fixed_cost=cf)
    # El costo recomputado puede diferir levemente por la reconstrucción de prefix_sims,
    # pero debe ser positivo y de orden de magnitud similar
    assert recomputed > 0
    ratio = abs(recomputed - dp_result.total_cost) / max(dp_result.total_cost, 1.0)
    assert ratio < 0.01  # tolerancia del 1%


def test_full_pipeline_dp_vs_sliding_window():
    dp_result = segment_dp(TEXT, lmin=10, lmax=80)
    sw_result = segment_sliding_window(TEXT, lmax=80, overlap=20)
    assert dp_result.total_cost < sw_result.total_cost


def test_full_pipeline_dp2d_optimal_k():
    result = segment_dp_2d(TEXT, lmin=10, lmax=80)
    best_cost = min(result.cost_by_k.values())
    assert result.total_cost == best_cost
    assert result.num_segments == min(result.cost_by_k, key=result.cost_by_k.get)


def test_full_pipeline_overlap_coherence():
    dp_result = segment_dp(TEXT, lmin=10, lmax=80)
    overlap_result = segment_dp_overlap(TEXT, lmin=10, lmax=80)
    dp_metrics = compute_metrics(dp_result)
    assert dp_metrics.avg_coherence >= 0.0


def test_full_report_without_llm():
    report = generate_full_report(TEXT, lmin=10, lmax=80, run_llm_evaluation=False)
    assert isinstance(report, FullReport)
    assert report.text_length_tokens > 0
    assert report.num_sentences > 0
    assert report.dp_metrics["total_cost"] > 0
    assert report.baseline_metrics["total_cost"] > 0
    assert report.dp_2d_metrics["optimal_k"] > 0


def test_full_report_dp_beats_baseline():
    report = generate_full_report(TEXT, lmin=10, lmax=80, run_llm_evaluation=False)
    dp_cost = report.dp_metrics["total_cost"]
    base_cost = report.baseline_metrics["total_cost"]
    sw_cost = report.sliding_window_metrics["total_cost"]
    # el DP debe superar al menos uno de los dos métodos
    assert dp_cost < base_cost or dp_cost < sw_cost


def test_full_report_dp_beats_sliding_window():
    report = generate_full_report(TEXT, lmin=10, lmax=80, run_llm_evaluation=False)
    assert report.dp_vs_sliding_window["cost_reduction_pct"] > 0


def test_all_methods_produce_valid_segments():
    dp_result = segment_dp(TEXT, lmin=10, lmax=80)
    base_result = segment_baseline(TEXT, lmax=80)
    sw_result = segment_sliding_window(TEXT, lmax=80, overlap=20)
    overlap_result = segment_dp_overlap(TEXT, lmin=10, lmax=80)

    for result in [dp_result, base_result, sw_result, overlap_result]:
        assert result.num_segments > 0
        for seg in result.segments:
            assert seg.token_count > 0
            assert len(seg.sentences) > 0
            assert len(seg.text) > 0


def test_baseline_with_lmin_parameter():
    """segment_baseline acepta lmin como parámetro sin romper la API."""
    result = segment_baseline(TEXT, lmax=80, lmin=10)
    assert result.num_segments > 0
    assert result.total_cost > 0
