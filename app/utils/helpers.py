from app.prompts import REFUSAL_KEYWORDS


def is_probable_refusal(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in REFUSAL_KEYWORDS)
