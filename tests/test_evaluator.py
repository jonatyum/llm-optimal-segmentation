import pytest
from unittest.mock import patch
from src.segmentation.evaluator import (
    evaluate_segmentation,
    compare_evaluations,
    EvaluationReport,
    SegmentEvaluation,
)
from src.segmentation.llm_client import InferenceResult

SEGMENTS = [
    "Large language models require significant computational resources for inference.",
    "Techniques like quantization help reduce inference costs substantially.",
]

_MOCK_RESULT = InferenceResult(
    prompt="test",
    response="This is a test response about language models and inference.",
    model="gemma2:2b",
    latency_ms=150.0,
    prompt_tokens=12,
    response_tokens=10,
    total_tokens=22,
)


@pytest.fixture(autouse=True)
def mock_ollama():
    with patch("src.segmentation.llm_client.run_inference", return_value=_MOCK_RESULT):
        yield


def test_evaluate_segmentation_returns_correct_type():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    assert isinstance(report, EvaluationReport)


def test_evaluate_segmentation_num_segments():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    assert report.num_segments == len(SEGMENTS)


def test_evaluate_segmentation_latency_is_positive():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    assert report.total_latency_ms > 0
    assert report.avg_latency_ms > 0


def test_evaluate_segmentation_tokens_are_positive():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    assert report.total_prompt_tokens > 0
    assert report.total_response_tokens > 0


def test_evaluate_segmentation_coherence_in_range():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    assert 0.0 <= report.avg_response_coherence <= 1.0


def test_segment_evaluations_length():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    assert len(report.segment_evaluations) == len(SEGMENTS)


def test_segment_evaluation_type():
    report = evaluate_segmentation(SEGMENTS, method="dp")
    for seg_eval in report.segment_evaluations:
        assert isinstance(seg_eval, SegmentEvaluation)


def test_compare_evaluations_returns_dict():
    dp_report = evaluate_segmentation(SEGMENTS, method="dp")
    base_report = evaluate_segmentation(SEGMENTS, method="baseline")
    comparison = compare_evaluations(dp_report, base_report)
    assert isinstance(comparison, dict)


def test_compare_evaluations_has_required_keys():
    dp_report = evaluate_segmentation(SEGMENTS, method="dp")
    base_report = evaluate_segmentation(SEGMENTS, method="baseline")
    comparison = compare_evaluations(dp_report, base_report)
    required_keys = [
        "latency_reduction_pct",
        "token_reduction_pct",
        "coherence_improvement",
        "dp_total_latency_ms",
        "baseline_total_latency_ms",
        "dp_avg_coherence",
        "baseline_avg_coherence",
    ]
    for key in required_keys:
        assert key in comparison