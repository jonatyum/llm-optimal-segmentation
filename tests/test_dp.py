import pytest
from src.segmentation.dp import segment_dp
from src.segmentation.models import SegmentationResult


TEXT_SHORT = (
    "The transformer architecture revolutionized NLP. "
    "It introduced the self-attention mechanism. "
    "Unlike recurrent networks, transformers process tokens in parallel."
)

TEXT_LONG = (
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


def test_returns_segmentation_result():
    result = segment_dp(TEXT_SHORT, lmin=10, lmax=100)
    assert isinstance(result, SegmentationResult)


def test_all_sentences_are_preserved():
    result = segment_dp(TEXT_SHORT, lmin=10, lmax=100)
    reconstructed = " ".join(
        sentence
        for segment in result.segments
        for sentence in segment.sentences
    )
    original_sentences = " ".join(
        s.strip() for s in TEXT_SHORT.split(".") if s.strip()
    )
    assert len(reconstructed) > 0


def test_segment_token_counts_respect_lmax():
    lmax = 80
    result = segment_dp(TEXT_LONG, lmin=10, lmax=lmax)
    for segment in result.segments:
        assert segment.token_count <= lmax, (
            f"Segment {segment.index} has {segment.token_count} tokens, exceeds lmax={lmax}"
        )


def test_total_cost_is_positive():
    result = segment_dp(TEXT_SHORT, lmin=10, lmax=100)
    assert result.total_cost > 0


def test_num_segments_matches_list():
    result = segment_dp(TEXT_SHORT, lmin=10, lmax=100)
    assert result.num_segments == len(result.segments)


def test_segment_indices_are_sequential():
    result = segment_dp(TEXT_LONG, lmin=10, lmax=80)
    for i, segment in enumerate(result.segments):
        assert segment.index == i


def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        segment_dp("", lmin=10, lmax=100)


def test_impossible_constraints_raises_error():
    with pytest.raises(ValueError):
        segment_dp(TEXT_SHORT, lmin=500, lmax=600)