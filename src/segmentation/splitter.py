# splitter.py — detección de oraciones con spaCy
import spacy

_nlp = None  # singleton — se carga una sola vez
_CHUNK_SIZE = 800_000  # máx chars por chunk (spaCy limita a 1M por defecto)


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _split_into_chunks(text: str) -> list[str]:
    """Divide texto largo en chunks respetando límite de spaCy, cortando en párrafos."""
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Preferir corte en párrafo → salto de línea → punto
        for sep in ('\n\n', '\n', '. ', ' '):
            pos = text.rfind(sep, start, end)
            if pos > start:
                end = pos + len(sep)
                break
        chunks.append(text[start:end])
        start = end
    return chunks


def split_sentences(text: str) -> list[str]:
    nlp = _get_nlp()
    text = text.strip()
    sentences = []
    for chunk in _split_into_chunks(text):
        doc = nlp(chunk)
        sentences.extend(s.text.strip() for s in doc.sents if s.text.strip())
    return sentences