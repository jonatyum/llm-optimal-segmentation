import pytest
from src.segmentation.baseline import segment_baseline
from src.segmentation.models import SegmentationResult


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


def test_returns_segmentation_result():
    result = segment_baseline(TEXT, lmax=100)
    assert isinstance(result, SegmentationResult)


def test_segment_token_counts_respect_lmax():
    lmax = 80
    result = segment_baseline(TEXT, lmax=lmax)
    for segment in result.segments:
        assert segment.token_count <= lmax, (
            f"Segment {segment.index} has {segment.token_count} tokens, exceeds lmax={lmax}"
        )


def test_total_cost_is_positive():
    result = segment_baseline(TEXT, lmax=100)
    assert result.total_cost > 0


def test_num_segments_matches_list():
    result = segment_baseline(TEXT, lmax=100)
    assert result.num_segments == len(result.segments)


def test_segment_indices_are_sequential():
    result = segment_baseline(TEXT, lmax=80)
    for i, segment in enumerate(result.segments):
        assert segment.index == i


def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        segment_baseline("", lmax=100)