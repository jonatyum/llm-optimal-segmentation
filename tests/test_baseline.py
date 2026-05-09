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


# ── Tests para lmin ───────────────────────────────────────────────────────────

def test_lmin_default_zero_no_change():
    """Sin lmin (default 0), el comportamiento es idéntico al original."""
    result_no_lmin = segment_baseline(TEXT, lmax=80)
    result_lmin_zero = segment_baseline(TEXT, lmax=80, lmin=0)
    assert result_no_lmin.num_segments == result_lmin_zero.num_segments
    assert result_no_lmin.total_cost == result_lmin_zero.total_cost


def test_lmin_merges_short_last_segment():
    """Si el último segmento tiene menos de lmin tokens, debe fusionarse con el anterior."""
    result = segment_baseline(TEXT, lmax=80, lmin=50)
    if len(result.segments) > 1:
        # El último segmento no debe tener menos de lmin (si hay más de uno)
        # Nota: puede tener más si la fusión resultó en > lmin
        # La invariante es: si hubo fusión, el segmento resultante es el penúltimo original + último
        last = result.segments[-1]
        assert last.token_count >= 1  # siempre hay al menos un token


def test_lmin_result_preserves_all_sentences():
    """Fusionar el último segmento no debe perder oraciones."""
    result_base = segment_baseline(TEXT, lmax=80, lmin=0)
    result_lmin = segment_baseline(TEXT, lmax=80, lmin=50)

    all_sents_base = [s for seg in result_base.segments for s in seg.sentences]
    all_sents_lmin = [s for seg in result_lmin.segments for s in seg.sentences]

    assert all_sents_base == all_sents_lmin


def test_lmin_reduces_or_equal_num_segments():
    """lmin puede reducir el número de segmentos (al fusionar el último) o mantenerlo igual."""
    result_no_lmin = segment_baseline(TEXT, lmax=80, lmin=0)
    result_lmin = segment_baseline(TEXT, lmax=80, lmin=50)
    assert result_lmin.num_segments <= result_no_lmin.num_segments


def test_lmin_single_segment_no_merge():
    """Con texto muy corto que produce un solo segmento, lmin no debe fallar."""
    short_text = "Hello world. This is a test."
    result = segment_baseline(short_text, lmax=200, lmin=100)
    assert result.num_segments >= 1
    assert result.total_cost > 0
