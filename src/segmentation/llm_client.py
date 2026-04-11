import time
import ollama
from dataclasses import dataclass

DEFAULT_MODEL = "gemma2:2b"


@dataclass
class InferenceResult:
    prompt: str
    response: str
    model: str
    latency_ms: float
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


def run_inference(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "You are a helpful assistant.",
) -> InferenceResult:
    start = time.perf_counter()

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    latency_ms = (time.perf_counter() - start) * 1000

    message = response.message.content
    prompt_tokens = response.prompt_eval_count or 0
    response_tokens = response.eval_count or 0

    return InferenceResult(
        prompt=prompt,
        response=message,
        model=model,
        latency_ms=round(latency_ms, 2),
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        total_tokens=prompt_tokens + response_tokens,
    )


def run_segmented_inference(
    segments: list[str],
    model: str = DEFAULT_MODEL,
    system_prompt: str = "You are a helpful assistant.",
) -> list[InferenceResult]:
    results = []
    for segment in segments:
        result = run_inference(
            prompt=segment,
            model=model,
            system_prompt=system_prompt,
        )
        results.append(result)
    return results