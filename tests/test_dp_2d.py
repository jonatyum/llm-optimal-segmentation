import pytest
from src.segmentation.dp import segment_dp_2d, DP2DResult


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


def test_returns_dp2d_result():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    assert isinstance(result, DP2DResult)


def test_cost_by_k_is_not_empty():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    assert len(result.cost_by_k) > 0


def test_optimal_k_has_minimum_cost():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    best_cost = min(result.cost_by_k.values())
    assert result.total_cost == best_cost


def test_costs_decrease_then_increase():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    costs = list(result.cost_by_k.values())
    min_idx = costs.index(min(costs))
    assert min_idx > 0


def test_num_segments_matches_best_k():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    assert result.num_segments == len(result.segments)


def test_segment_indices_are_sequential():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    for i, segment in enumerate(result.segments):
        assert segment.index == i


def test_segment_token_counts_respect_lmax():
    lmax = 80
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=lmax)
    for segment in result.segments:
        assert segment.token_count <= lmax


def test_total_cost_is_positive():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100)
    assert result.total_cost > 0


def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        segment_dp_2d("", lmin=10, lmax=100)


def test_impossible_constraints_raises_error():
    with pytest.raises(ValueError):
        segment_dp_2d(TEXT_SHORT, lmin=500, lmax=600)


def test_max_k_limits_search():
    result = segment_dp_2d(TEXT_LONG, lmin=10, lmax=100, max_k=3)
    assert max(result.cost_by_k.keys()) <= 3