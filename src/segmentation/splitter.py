# splitter.py — detección de oraciones con spaCy
import spacy

_nlp = None  # singleton — se carga una sola vez


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# divide texto en oraciones — unidades atómicas del DP
def split_sentences(text: str) -> list[str]:
    nlp = _get_nlp()
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]