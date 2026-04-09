import tiktoken

SUPPORTED_MODELS = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

DEFAULT_MODEL = "gpt-4o"


def get_encoder(model: str = DEFAULT_MODEL) -> tiktoken.Encoding:
    encoding_name = SUPPORTED_MODELS.get(model)
    if not encoding_name:
        raise ValueError(f"Model '{model}' not supported. Choose from: {list(SUPPORTED_MODELS)}")
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    encoder = get_encoder(model)
    return len(encoder.encode(text))


def count_tokens_batch(texts: list[str], model: str = DEFAULT_MODEL) -> list[int]:
    encoder = get_encoder(model)
    return [len(encoder.encode(t)) for t in texts]