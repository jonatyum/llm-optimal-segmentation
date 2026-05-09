import pytest
from src.segmentation.texttiling import segment_texttiling
from src.segmentation.models import SegmentationResult, Segment


# Multi-topic text with clear thematic breaks
MULTI_TOPIC_TEXT = (
    "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 CE. "
    "Roman roads connected distant provinces and enabled rapid troop movement. "
    "The Senate debated policy while the legions guarded frontiers stretching thousands of miles. "
    "Photosynthesis is the process by which plants convert light energy into chemical energy. "
    "Chlorophyll in the chloroplasts absorbs sunlight and drives the splitting of water molecules. "
    "The Calvin cycle fixes atmospheric carbon dioxide into glucose using the energy produced. "
    "Market equilibrium occurs when the quantity supplied equals the quantity demanded at a given price. "
    "When prices rise above equilibrium, suppliers produce more but consumers buy less, creating a surplus. "
    "Competition among sellers then drives prices back down toward the equilibrium level."
)

SHORT_TEXT = (
    "Neural networks consist of layers of interconnected nodes. "
    "Each node applies a nonlinear activation function to a weighted sum of its inputs."
)

SINGLE_SENTENCE = "The mitochondrion is the powerhouse of the cell."

LONG_TEXT = (
    "Quantum entanglement describes a phenomenon where two particles become correlated such that "
    "measuring the state of one instantly determines the state of the other, regardless of distance. "
    "Einstein famously called this spooky action at a distance and was deeply skeptical of the interpretation. "
    "Bell's theorem later provided a testable inequality that distinguishes quantum mechanics from "
    "local hidden variable theories, and experiments have consistently confirmed quantum predictions. "
    "The Amazon rainforest contains approximately ten percent of all species on Earth. "
    "Deforestation driven by agricultural expansion has eliminated more than seventeen percent "
    "of the original forest cover since systematic monitoring began in the 1970s. "
    "The forest acts as a critical carbon sink, absorbing billions of tonnes of carbon dioxide annually. "
    "Keynesian economics argues that aggregate demand is the primary driver of output and employment. "
    "During recessions, the government should increase spending to offset the collapse in private demand. "
    "The multiplier effect means each unit of government expenditure generates more than one unit "
    "of additional income through successive rounds of consumer spending."
)


def test_returns_segmentation_result():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    assert isinstance(result, SegmentationResult)


def test_segments_are_segment_instances():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    for seg in result.segments:
        assert isinstance(seg, Segment)


def test_num_segments_matches_list_length():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    assert result.num_segments == len(result.segments)


def test_segment_indices_are_sequential():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    for i, seg in enumerate(result.segments):
        assert seg.index == i


def test_all_sentences_preserved():
    """No sentence should be lost during segmentation."""
    from src.segmentation.splitter import split_sentences
    original_sentences = split_sentences(MULTI_TOPIC_TEXT)
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    segmented_sentences = [s for seg in result.segments for s in seg.sentences]
    assert segmented_sentences == original_sentences


def test_lmax_constraint_respected():
    lmax = 80
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=10, lmax=lmax)
    for seg in result.segments:
        assert seg.token_count <= lmax, (
            f"Segment {seg.index} has {seg.token_count} tokens, exceeds lmax={lmax}"
        )


def test_single_sentence_returns_one_segment():
    result = segment_texttiling(SINGLE_SENTENCE, lmin=1, lmax=200)
    assert result.num_segments == 1
    assert result.segments[0].token_count > 0


def test_no_empty_segments():
    result = segment_texttiling(LONG_TEXT, lmin=20, lmax=150)
    for seg in result.segments:
        assert seg.sentences, f"Segment {seg.index} has no sentences"
        assert seg.token_count > 0, f"Segment {seg.index} has zero tokens"
        assert seg.text.strip(), f"Segment {seg.index} has empty text"


def test_total_cost_is_sum_of_squared_token_counts():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    expected_cost = sum(s.token_count ** 2 for s in result.segments)
    assert abs(result.total_cost - expected_cost) < 1e-6


def test_empty_text_raises_value_error():
    with pytest.raises(ValueError):
        segment_texttiling("", lmin=20, lmax=120)


def test_short_text_produces_at_least_one_segment():
    result = segment_texttiling(SHORT_TEXT, lmin=5, lmax=200)
    assert result.num_segments >= 1
    assert result.total_cost > 0


def test_window_size_one_still_works():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120, window_size=1)
    assert result.num_segments >= 1
    assert all(s.token_count > 0 for s in result.segments)


def test_smoothing_passes_zero_still_works():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120, smoothing_passes=0)
    assert result.num_segments >= 1
    assert all(s.token_count > 0 for s in result.segments)


def test_segment_text_matches_joined_sentences():
    result = segment_texttiling(MULTI_TOPIC_TEXT, lmin=20, lmax=120)
    for seg in result.segments:
        expected_text = " ".join(seg.sentences)
        assert seg.text == expected_text
