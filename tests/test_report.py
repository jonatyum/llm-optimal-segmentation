# tests/test_report.py
import pytest
from src.segmentation.report import generate_full_report, FullReport

TEXT = (
    "Large language models have billions of parameters. "
    "They require significant computational resources for inference. "
    "Techniques like quantization help reduce inference costs. "
    "Prompt engineering has emerged as a key skill. "
    "Retrieval-augmented generation combines LLMs with external knowledge. "
    "Fine-tuning on domain-specific data improves performance. "
    "Evaluation remains a challenge for these models. "
    "Automated metrics often fail to capture true quality. "
    "Human evaluation is expensive but more reliable. "
    "Benchmarks like MMLU and HellaSwag are widely used."
)


def test_generate_full_report_returns_correct_type():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    assert isinstance(report, FullReport)


def test_report_text_length_is_positive():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    assert report.text_length_tokens > 0


def test_report_num_sentences_is_positive():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    assert report.num_sentences > 0


def test_report_dp_metrics_has_required_keys():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    required_keys = [
        "num_segments", "total_cost", "avg_tokens",
        "std_tokens", "avg_coherence", "llm_evaluation"
    ]
    for key in required_keys:
        assert key in report.dp_metrics


def test_report_baseline_metrics_has_required_keys():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    required_keys = [
        "num_segments", "total_cost", "avg_tokens",
        "std_tokens", "avg_coherence", "llm_evaluation"
    ]
    for key in required_keys:
        assert key in report.baseline_metrics


def test_report_dp_vs_baseline_has_required_keys():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    required_keys = [
        "cost_reduction_pct", "coherence_improvement", "segment_diff"
    ]
    for key in required_keys:
        assert key in report.dp_vs_baseline


def test_report_dp_vs_sliding_window_has_required_keys():
    report = generate_full_report(TEXT, lmin=10, lmax=100)
    required_keys = [
        "cost_reduction_pct", "segment_diff", "overlap_token_diff"
    ]
    for key in required_keys:
        assert key in report.dp_vs_sliding_window


def test_report_without_llm_evaluation():
    report = generate_full_report(TEXT, lmin=10, lmax=100, run_llm_evaluation=False)
    assert report.dp_metrics["llm_evaluation"] == {}
    assert report.baseline_metrics["llm_evaluation"] == {}
    assert report.sliding_window_metrics["llm_evaluation"] == {}