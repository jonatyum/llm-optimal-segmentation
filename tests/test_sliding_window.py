import pytest
from src.segmentation.sliding_window import (
    segment_sliding_window,
    SlidingWindowResult,
    SlidingWindowSegment,
)

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


def test_returns_sliding_window_result():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    assert isinstance(result, SlidingWindowResult)


def test_num_segments_matches_list():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    assert result.num_segments == len(result.segments)


def test_segment_indices_are_sequential():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    for i, segment in enumerate(result.segments):
        assert segment.index == i


def test_segment_token_counts_respect_lmax():
    lmax = 80
    result = segment_sliding_window(TEXT, lmax=lmax, overlap=20)
    for segment in result.segments:
        assert segment.token_count <= lmax, (
            f"Segment {segment.index} has {segment.token_count} tokens, exceeds lmax={lmax}"
        )


def test_total_cost_is_positive():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    assert result.total_cost > 0


def test_overlap_tokens_are_non_negative():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    for segment in result.segments:
        assert segment.overlap_tokens >= 0


def test_first_segment_has_no_overlap():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    assert result.segments[0].overlap_tokens == 0


def test_total_overlap_tokens_is_non_negative():
    result = segment_sliding_window(TEXT, lmax=100, overlap=20)
    assert result.total_overlap_tokens >= 0


def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        segment_sliding_window("", lmax=100, overlap=20)


def test_overlap_greater_than_lmax_raises_error():
    with pytest.raises(ValueError):
        segment_sliding_window(TEXT, lmax=100, overlap=150)


def test_zero_overlap_produces_no_overlap():
    result = segment_sliding_window(TEXT, lmax=100, overlap=0)
    for segment in result.segments:
        assert segment.overlap_tokens == 0