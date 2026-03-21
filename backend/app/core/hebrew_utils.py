import re
import unicodedata

# Hebrew nikud (vowel points) Unicode range: U+0591 to U+05C7
NIKUD_PATTERN = re.compile(r"[\u0591-\u05C7]")
# Hebrew punctuation: maqaf, paseq, sof pasuq, nun hafukha, geresh, gershayim
HEBREW_PUNCT_PATTERN = re.compile(r"[\u05BE\u05C0\u05C3\u05C6\u05F3\u05F4]")


def strip_nikud(text: str) -> str:
    return NIKUD_PATTERN.sub("", text)


def normalize_hebrew(text: str) -> str:
    """Normalize Hebrew text for fair WER comparison."""
    text = unicodedata.normalize("NFC", text)
    text = strip_nikud(text)
    text = HEBREW_PUNCT_PATTERN.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
