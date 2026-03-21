import json

import jiwer

from app.core.hebrew_utils import normalize_hebrew


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_hebrew(reference)
    hyp = normalize_hebrew(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    return jiwer.wer(ref, hyp)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_hebrew(reference)
    hyp = normalize_hebrew(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    return jiwer.cer(ref, hyp)


def compute_word_diff(reference: str, hypothesis: str) -> list[dict]:
    """
    Compute word-level alignment between reference and hypothesis.
    Returns list of {ref_word, hyp_word, status} where status is
    "correct", "substitution", "insertion", "deletion".
    """
    ref = normalize_hebrew(reference)
    hyp = normalize_hebrew(hypothesis)

    if not ref and not hyp:
        return []

    if not ref:
        return [
            {"ref_word": None, "hyp_word": w, "status": "insertion"}
            for w in hyp.split()
        ]

    if not hyp:
        return [
            {"ref_word": w, "hyp_word": None, "status": "deletion"}
            for w in ref.split()
        ]

    output = jiwer.process_words(ref, hyp)
    ref_words = ref.split()
    hyp_words = hyp.split()

    diff = []
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                diff.append({
                    "ref_word": ref_words[chunk.ref_start_idx + i],
                    "hyp_word": hyp_words[chunk.hyp_start_idx + i],
                    "status": "correct",
                })
        elif chunk.type == "substitute":
            ref_count = chunk.ref_end_idx - chunk.ref_start_idx
            hyp_count = chunk.hyp_end_idx - chunk.hyp_start_idx
            for i in range(max(ref_count, hyp_count)):
                r = ref_words[chunk.ref_start_idx + i] if i < ref_count else None
                h = hyp_words[chunk.hyp_start_idx + i] if i < hyp_count else None
                if r and h:
                    diff.append({"ref_word": r, "hyp_word": h, "status": "substitution"})
                elif r:
                    diff.append({"ref_word": r, "hyp_word": None, "status": "deletion"})
                else:
                    diff.append({"ref_word": None, "hyp_word": h, "status": "insertion"})
        elif chunk.type == "delete":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                diff.append({
                    "ref_word": ref_words[chunk.ref_start_idx + i],
                    "hyp_word": None,
                    "status": "deletion",
                })
        elif chunk.type == "insert":
            for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                diff.append({
                    "ref_word": None,
                    "hyp_word": hyp_words[chunk.hyp_start_idx + i],
                    "status": "insertion",
                })

    return diff


def word_diff_to_json(diff: list[dict]) -> str:
    return json.dumps(diff, ensure_ascii=False)
