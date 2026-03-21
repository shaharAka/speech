"""WER evaluation with Hebrew normalization."""

import jiwer

from app.core.hebrew_utils import normalize_hebrew


def compute_wer_hebrew(predictions: list[str], references: list[str]) -> float:
    """Compute WER with Hebrew text normalization."""
    norm_preds = [normalize_hebrew(p) for p in predictions]
    norm_refs = [normalize_hebrew(r) for r in references]

    # Filter out empty pairs
    pairs = [(r, p) for r, p in zip(norm_refs, norm_preds) if r.strip()]
    if not pairs:
        return 0.0

    refs, preds = zip(*pairs)
    return jiwer.wer(list(refs), list(preds))
