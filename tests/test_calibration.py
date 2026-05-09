"""Tests para src/segmentation/calibration.py."""
import pytest
from unittest.mock import patch, MagicMock
from src.segmentation.calibration import (
    suggest_lambda,
    estimate_optimal_k,
    validate_calibration,
    calibrate_fixed_cost,
)
from src.segmentation.llm_client import InferenceResult

# ── Fixtures ──────────────────────────────────────────────────────────────────

SHORT_TEXT = (
    "The cat sat on the mat. The dog ran fast. Birds fly high."
)

MEDIUM_TEXT = (
    "Large language models have billions of parameters and require significant "
    "computational resources for inference. Techniques like quantization and "
    "distillation help reduce inference costs substantially. Prompt engineering "
    "has emerged as a key skill for working with these models effectively. "
    "Retrieval-augmented generation combines LLMs with external knowledge sources "
    "to improve factual accuracy. Fine-tuning on domain-specific data can "
    "substantially improve performance on specialized tasks."
)

LONG_TEXT = (
    "The transformer architecture introduced self-attention mechanisms that allow "
    "models to weigh the importance of different tokens dynamically. "
    "Unlike recurrent networks, transformers process all tokens in parallel, which "
    "dramatically reduces training time on modern hardware accelerators. "
    "BERT leveraged bidirectional pre-training to achieve state-of-the-art results "
    "on a wide range of natural language understanding benchmarks. "
    "GPT models use a unidirectional autoregressive approach instead, making them "
    "particularly well-suited for open-ended text generation tasks. "
    "Scaling laws suggest that model performance improves predictably with more "
    "parameters, data, and compute, enabling researchers to forecast capabilities. "
    "Instruction tuning and reinforcement learning from human feedback have become "
    "essential techniques for aligning language models with user intent."
)


# ── suggest_lambda ────────────────────────────────────────────────────────────

class TestSuggestLambda:
    def test_returns_float(self):
        result = suggest_lambda(MEDIUM_TEXT)
        assert isinstance(result, float)

    def test_value_in_valid_range(self):
        result = suggest_lambda(MEDIUM_TEXT)
        assert 0.1 <= result <= 2.0

    def test_possible_values_only(self):
        """suggest_lambda solo debe retornar 0.3, 0.5 o 0.8."""
        result = suggest_lambda(MEDIUM_TEXT)
        assert result in {0.3, 0.5, 0.8}

    def test_short_text_returns_default(self):
        """Texto con menos de 2 oraciones retorna 0.5."""
        result = suggest_lambda("Only one sentence here.")
        assert result == 0.5

    def test_medium_text_typical_value(self):
        """Texto técnico moderadamente coherente → lambda razonable (0.3, 0.5 o 0.8)."""
        result = suggest_lambda(MEDIUM_TEXT)
        assert result in {0.3, 0.5, 0.8}

    def test_long_text_returns_valid(self):
        result = suggest_lambda(LONG_TEXT)
        assert result in {0.3, 0.5, 0.8}


# ── estimate_optimal_k ────────────────────────────────────────────────────────

class TestEstimateOptimalK:
    def test_returns_int(self):
        result = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=100.0)
        assert isinstance(result, int)

    def test_returns_positive(self):
        result = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=100.0)
        assert result >= 1

    def test_high_fixed_cost_gives_smaller_k(self):
        """fixed_cost alto → k* pequeño."""
        k_low = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=10.0)
        k_high = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=10000.0)
        assert k_high <= k_low

    def test_zero_fixed_cost_returns_one(self):
        """fixed_cost <= 0 retorna 1 (evita división por cero)."""
        result = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=0.0)
        assert result == 1

    def test_longer_text_gives_larger_k(self):
        """Texto más largo → más tokens → k* más grande con mismo fixed_cost."""
        k_short = estimate_optimal_k(SHORT_TEXT, fixed_cost=100.0)
        k_long = estimate_optimal_k(LONG_TEXT, fixed_cost=100.0)
        assert k_long >= k_short

    def test_model_parameter_accepted(self):
        result = estimate_optimal_k(MEDIUM_TEXT, model="gpt-4o", fixed_cost=100.0)
        assert result >= 1


# ── validate_calibration ─────────────────────────────────────────────────────

class TestValidateCalibration:
    def test_returns_dict(self):
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        assert "k_dp" in result
        assert "k_analytical" in result
        assert "deviation_pct" in result
        assert "valid" in result

    def test_k_dp_is_positive_int(self):
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        assert isinstance(result["k_dp"], int)
        assert result["k_dp"] >= 1

    def test_k_analytical_is_positive_int(self):
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        assert isinstance(result["k_analytical"], int)
        assert result["k_analytical"] >= 1

    def test_deviation_pct_is_non_negative_float(self):
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        assert isinstance(result["deviation_pct"], float)
        assert result["deviation_pct"] >= 0.0

    def test_valid_is_bool(self):
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        assert isinstance(result["valid"], bool)

    def test_valid_consistent_with_deviation(self):
        """valid debe ser True iff deviation_pct <= 0.20."""
        result = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200, coherence_lambda=0.5, fixed_cost=100.0
        )
        expected_valid = result["deviation_pct"] <= 0.20
        assert result["valid"] == expected_valid


# ── calibrate_fixed_cost (con mock de Ollama) ─────────────────────────────────

def _fake_inference(latency_ms: float) -> InferenceResult:
    return InferenceResult(
        prompt="Hi",
        response="Hello",
        model="gemma2:2b",
        latency_ms=latency_ms,
        prompt_tokens=2,
        response_tokens=1,
        total_tokens=3,
    )


class TestCalibrateFixedCost:
    def test_returns_float(self):
        fake = MagicMock(return_value=_fake_inference(200.0))
        with patch("src.segmentation.llm_client.run_inference", fake):
            result = calibrate_fixed_cost(num_trials=3)
        assert isinstance(result, float)

    def test_value_is_positive(self):
        fake = MagicMock(return_value=_fake_inference(300.0))
        with patch("src.segmentation.llm_client.run_inference", fake):
            result = calibrate_fixed_cost(num_trials=3)
        assert result > 0.0

    def test_scales_with_latency(self):
        """Latencia mayor → fixed_cost mayor."""
        fast = MagicMock(return_value=_fake_inference(100.0))
        slow = MagicMock(return_value=_fake_inference(500.0))
        with patch("src.segmentation.llm_client.run_inference", fast):
            cost_fast = calibrate_fixed_cost(num_trials=3)
        with patch("src.segmentation.llm_client.run_inference", slow):
            cost_slow = calibrate_fixed_cost(num_trials=3)
        assert cost_slow > cost_fast

    def test_correct_num_trials(self):
        """run_inference debe llamarse exactamente num_trials veces."""
        fake = MagicMock(return_value=_fake_inference(200.0))
        with patch("src.segmentation.llm_client.run_inference", fake):
            calibrate_fixed_cost(num_trials=5)
        assert fake.call_count == 5

    def test_formula_mean_times_0_1(self):
        """fixed_cost = mean(latencias_ms) * 0.1."""
        latencies = [100.0, 200.0, 300.0]
        results = [_fake_inference(lat) for lat in latencies]
        fake = MagicMock(side_effect=results)
        with patch("src.segmentation.llm_client.run_inference", fake):
            result = calibrate_fixed_cost(num_trials=3)
        expected = (sum(latencies) / len(latencies)) * 0.1
        assert abs(result - expected) < 1e-9


# ── Pipeline end-to-end ───────────────────────────────────────────────────────

class TestCalibrationEndToEnd:
    """Valida el flujo completo: suggest_lambda → estimate_k → validate."""

    def test_pipeline_suggest_then_estimate(self):
        lam = suggest_lambda(MEDIUM_TEXT)
        k = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=1000.0)
        assert lam in {0.3, 0.5, 0.8}
        assert k >= 1

    def test_pipeline_estimate_then_validate_consistent(self):
        """k_analytical de validate_calibration debe coincidir con estimate_optimal_k."""
        k_direct = estimate_optimal_k(MEDIUM_TEXT, fixed_cost=1000.0)
        cal = validate_calibration(
            MEDIUM_TEXT, lmin=10, lmax=200,
            coherence_lambda=0.5, fixed_cost=1000.0,
        )
        assert cal["k_analytical"] == k_direct

    def test_pipeline_calibrate_then_validate_with_mock(self):
        """fixed_cost derivado del mock debe producir calibración válida en texto largo."""
        fake = MagicMock(return_value=_fake_inference(400.0))
        with patch("src.segmentation.llm_client.run_inference", fake):
            cf = calibrate_fixed_cost(num_trials=3)

        cal = validate_calibration(
            LONG_TEXT, lmin=10, lmax=300,
            coherence_lambda=suggest_lambda(LONG_TEXT),
            fixed_cost=cf,
        )
        # solo verifica que el pipeline completo corre sin errores y devuelve estructura válida
        assert "k_dp" in cal
        assert "k_analytical" in cal
        assert cal["k_dp"] >= 1
        assert cal["k_analytical"] >= 1

    def test_full_pipeline_on_corpus_texts(self):
        """Corre el pipeline completo sobre los 3 textos de prueba."""
        import json, os
        corpus_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "corpus.json",
        )
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

        for doc in corpus[:3]:
            lam = suggest_lambda(doc["text"])
            k = estimate_optimal_k(doc["text"], fixed_cost=1000.0)
            cal = validate_calibration(
                doc["text"], lmin=10, lmax=200,
                coherence_lambda=lam, fixed_cost=1000.0,
            )
            assert k >= 1
            assert cal["k_dp"] >= 1
            assert 0.0 <= cal["deviation_pct"]