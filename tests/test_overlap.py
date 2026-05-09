import pytest
from src.segmentation.overlap import segment_dp_overlap, OverlapSegmentationResult


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


def test_returns_overlap_segmentation_result():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100)
    assert isinstance(result, OverlapSegmentationResult)


def test_num_segments_matches_list():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100)
    assert result.num_segments == len(result.segments)


def test_segment_indices_are_sequential():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100)
    for i, segment in enumerate(result.segments):
        assert segment.index == i


def test_total_cost_is_positive():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100)
    assert result.total_cost > 0


def test_overlap_tokens_are_non_negative():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100)
    for segment in result.segments:
        assert segment.overlap_tokens >= 0


def test_total_overlap_tokens_is_non_negative():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100)
    assert result.total_overlap_tokens >= 0


def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        segment_dp_overlap("", lmin=10, lmax=100)


def test_impossible_constraints_raises_error():
    with pytest.raises(ValueError):
        segment_dp_overlap(TEXT, lmin=500, lmax=600)


def test_max_overlap_zero_produces_no_overlap():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100, max_overlap=0)
    for segment in result.segments:
        assert segment.overlap_tokens == 0
        assert segment.overlap_sentences == []


def test_overlap_produces_nonzero_tokens_at_coherent_boundaries():
    result = segment_dp_overlap(TEXT, lmin=10, lmax=100, overlap_mu=0.01, max_overlap=3)
    assert result.total_overlap_tokens > 0, (
        "Expected nonzero overlap when overlap_mu is low and text has coherent boundaries"
    )