# tests/test_metrics.py
import pytest
from src.segmentation.dp import segment_dp
from src.segmentation.baseline import segment_baseline
from src.segmentation.metrics import compute_metrics, compare, SegmentationMetrics, ComparisonReport


TEXT = (
    "Large language models have billions of parameters. "
    "They require significant computational resources for inference. "
    "Techniques like quantization help reduce inference costs. "
    "Prompt engineering has emerged as a key skill. "
    "Retrieval-augmented generation combines LLMs with external knowledge. "
    "Fine-tuning on domain-specific data improves performance. "
    "Evaluation remains a challenge for these models. "
    "Automated metrics often fail to capture true quality."
)


def test_compute_metrics_returns_correct_type():
    result = segment_dp(TEXT, lmin=10, lmax=100)
    metrics = compute_metrics(result)
    assert isinstance(metrics, SegmentationMetrics)


def test_compute_metrics_total_tokens_is_consistent():
    result = segment_dp(TEXT, lmin=10, lmax=100)
    metrics = compute_metrics(result)
    expected_total = sum(s.token_count for s in result.segments)
    assert metrics.total_tokens == expected_total


def test_compute_metrics_min_max_are_consistent():
    result = segment_dp(TEXT, lmin=10, lmax=100)
    metrics = compute_metrics(result)
    assert metrics.min_tokens <= metrics.avg_tokens_per_segment <= metrics.max_tokens


def test_compute_metrics_std_is_non_negative():
    result = segment_dp(TEXT, lmin=10, lmax=100)
    metrics = compute_metrics(result)
    assert metrics.std_tokens_per_segment >= 0


def test_compare_returns_correct_type():
    dp_result = segment_dp(TEXT, lmin=10, lmax=100)
    base_result = segment_baseline(TEXT, lmax=100)
    report = compare(dp_result, base_result)
    assert isinstance(report, ComparisonReport)


def test_compare_cost_reduction_is_valid_percentage():
    dp_result = segment_dp(TEXT, lmin=10, lmax=100)
    base_result = segment_baseline(TEXT, lmax=100)
    report = compare(dp_result, base_result)
    assert -100.0 <= report.cost_reduction_pct <= 100.0


def test_compare_segment_diff_is_integer():
    dp_result = segment_dp(TEXT, lmin=10, lmax=100)
    base_result = segment_baseline(TEXT, lmax=100)
    report = compare(dp_result, base_result)
    assert isinstance(report.segment_diff, int)