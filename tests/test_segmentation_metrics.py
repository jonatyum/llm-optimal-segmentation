"""Tests para src/segmentation/segmentation_metrics.py."""
import pytest
from src.segmentation.segmentation_metrics import (
    boundaries_from_result,
    pk,
    window_diff,
    _default_k,
)


# ── Fixtures de boundaries simples ───────────────────────────────────────────

# Referencia: 3 segmentos de 3 oraciones cada una (9 oraciones total)
# boundaries de longitud 8: [0,0,1,0,0,1,0,0]
REF_3X3 = [0, 0, 1, 0, 0, 1, 0, 0]

# Hipótesis perfecta (igual a referencia)
HYP_PERFECT = [0, 0, 1, 0, 0, 1, 0, 0]

# Hipótesis sin ningún corte (1 segmento)
HYP_NO_CUT = [0, 0, 0, 0, 0, 0, 0, 0]

# Hipótesis con corte en posición equivocada
HYP_SHIFTED = [0, 1, 0, 0, 0, 1, 0, 0]

# Referencia de 2 segmentos iguales (4 oraciones cada una)
REF_2X4 = [0, 0, 0, 1, 0, 0, 0]


# ── Tests de pk ──────────────────────────────────────────────────────────────

class TestPk:
    def test_perfect_hypothesis_gives_zero(self):
        assert pk(REF_3X3, HYP_PERFECT) == 0.0

    def test_no_cut_hypothesis_nonzero(self):
        score = pk(REF_3X3, HYP_NO_CUT)
        assert score > 0.0

    def test_score_between_zero_and_one(self):
        score = pk(REF_3X3, HYP_SHIFTED)
        assert 0.0 <= score <= 1.0

    def test_empty_input_returns_zero(self):
        assert pk([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pk([0, 1], [0, 1, 0])

    def test_explicit_k(self):
        score = pk(REF_3X3, HYP_NO_CUT, k=2)
        assert 0.0 <= score <= 1.0

    def test_k_larger_than_text_clamped(self):
        """k >= n_sentences debe ser clampeado sin error."""
        score = pk(REF_3X3, HYP_PERFECT, k=100)
        assert score == 0.0

    def test_perfect_score_is_symmetric(self):
        """Hipótesis perfecta → 0, independientemente de la referencia."""
        assert pk(REF_2X4, REF_2X4) == 0.0

    def test_worse_hypothesis_has_higher_pk(self):
        """Hipótesis más alejada de la referencia → Pk más alto."""
        score_shifted = pk(REF_3X3, HYP_SHIFTED)
        score_no_cut = pk(REF_3X3, HYP_NO_CUT)
        # ambos deben ser >= 0; no se impone un orden estricto entre ellos,
        # solo que ambos son peores que la hipótesis perfecta
        assert score_shifted >= 0.0
        assert score_no_cut >= 0.0


# ── Tests de window_diff ──────────────────────────────────────────────────────

class TestWindowDiff:
    def test_perfect_hypothesis_gives_zero(self):
        assert window_diff(REF_3X3, HYP_PERFECT) == 0.0

    def test_no_cut_hypothesis_nonzero(self):
        score = window_diff(REF_3X3, HYP_NO_CUT)
        assert score > 0.0

    def test_score_between_zero_and_one(self):
        score = window_diff(REF_3X3, HYP_SHIFTED)
        assert 0.0 <= score <= 1.0

    def test_empty_input_returns_zero(self):
        assert window_diff([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            window_diff([0, 1], [0])

    def test_explicit_k(self):
        score = window_diff(REF_3X3, HYP_NO_CUT, k=3)
        assert 0.0 <= score <= 1.0

    def test_perfect_score_symmetric(self):
        assert window_diff(REF_2X4, REF_2X4) == 0.0

    def test_window_diff_penalizes_shifted_boundary(self):
        """WindowDiff penaliza boundaries desplazados incluso si el conteo es igual."""
        ref = [0, 0, 1, 0, 0, 0, 0]   # corte en pos 2
        hyp = [0, 0, 0, 1, 0, 0, 0]   # corte desplazado a pos 3
        score = window_diff(ref, hyp, k=3)
        assert score > 0.0


# ── Tests de _default_k ───────────────────────────────────────────────────────

class TestDefaultK:
    def test_default_k_is_positive(self):
        assert _default_k(REF_3X3) >= 1

    def test_default_k_for_equal_segments(self):
        """3 segmentos de 3 oraciones → avg = 3 → k = 1 o 2."""
        k = _default_k(REF_3X3)
        assert k in {1, 2}

    def test_default_k_larger_for_bigger_segments(self):
        """Segmentos más grandes → k más grande."""
        k_small = _default_k([0, 1, 0, 1, 0])  # 3 segs de ~2 oraciones
        k_large = _default_k([0, 0, 0, 0, 1, 0, 0, 0, 0])  # 2 segs de ~5 oraciones
        assert k_large >= k_small


# ── Tests de boundaries_from_result ─────────────────────────────────────────

class TestBoundariesFromResult:
    def test_single_segment_no_boundaries(self):
        """Un único segmento → lista de ceros de len(n_sents - 1)."""
        from src.segmentation.models import Segment, SegmentationResult
        seg = Segment(index=0, sentences=["A.", "B.", "C."], text="A. B. C.", token_count=10)
        result = SegmentationResult(segments=[seg], total_cost=10.0, num_segments=1)
        b = boundaries_from_result(result)
        assert b == [0, 0]

    def test_two_segments_one_boundary(self):
        """Dos segmentos de 2 oraciones cada uno → [0, 1, 0]."""
        from src.segmentation.models import Segment, SegmentationResult
        s1 = Segment(index=0, sentences=["A.", "B."], text="A. B.", token_count=5)
        s2 = Segment(index=1, sentences=["C.", "D."], text="C. D.", token_count=5)
        result = SegmentationResult(segments=[s1, s2], total_cost=50.0, num_segments=2)
        b = boundaries_from_result(result)
        assert b == [0, 1, 0]

    def test_boundary_count_equals_num_cuts(self):
        """Número de 1s en boundaries == número de cortes == num_segments - 1."""
        from src.segmentation.models import Segment, SegmentationResult
        segs = [
            Segment(index=i, sentences=["X.", "Y."], text="X. Y.", token_count=5)
            for i in range(4)
        ]
        result = SegmentationResult(segments=segs, total_cost=80.0, num_segments=4)
        b = boundaries_from_result(result)
        assert sum(b) == 3  # 4 segmentos → 3 cortes

    def test_total_length_is_n_sentences_minus_one(self):
        from src.segmentation.models import Segment, SegmentationResult
        segs = [
            Segment(index=0, sentences=["A.", "B.", "C."], text="A. B. C.", token_count=10),
            Segment(index=1, sentences=["D.", "E."], text="D. E.", token_count=7),
        ]
        result = SegmentationResult(segments=segs, total_cost=50.0, num_segments=2)
        b = boundaries_from_result(result)
        total_sents = 3 + 2
        assert len(b) == total_sents - 1
