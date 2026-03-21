"""Adaptive text generation using Gemini — analyzes past performance to create targeted practice texts."""

import json
import logging
from collections import Counter

import google.generativeai as genai
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.recording import Recording
from app.models.text import Text
from app.models.transcription import Transcription

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


async def analyze_round_performance(round_num: int, db: AsyncSession) -> dict:
    """Analyze all transcription results from a given round to find error patterns."""

    # Get all text IDs in the round
    text_ids_result = await db.execute(
        select(Text.id, Text.difficulty).where(Text.round == round_num)
    )
    text_rows = text_ids_result.all()
    if not text_rows:
        return {"has_data": False}

    text_ids = [r[0] for r in text_rows]
    text_difficulties = {r[0]: r[1] for r in text_rows}

    # Get all transcriptions for recordings of these texts
    transcriptions_result = await db.execute(
        select(Transcription, Recording.text_id)
        .join(Recording, Transcription.recording_id == Recording.id)
        .where(Recording.text_id.in_(text_ids))
    )
    rows = transcriptions_result.all()

    if not rows:
        return {"has_data": False}

    # Aggregate stats
    wer_by_difficulty: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    failed_words: Counter = Counter()
    substitution_pairs: Counter = Counter()
    deletion_words: Counter = Counter()
    total_correct = 0
    total_errors = 0

    for transcription, text_id in rows:
        difficulty = text_difficulties.get(text_id, "medium")
        wer_by_difficulty[difficulty].append(transcription.wer_score)

        # Parse word diff
        try:
            word_diff = json.loads(transcription.word_diff_json)
        except (json.JSONDecodeError, TypeError):
            continue

        for entry in word_diff:
            status = entry.get("status", "")
            ref_word = entry.get("ref_word", "")
            hyp_word = entry.get("hyp_word", "")

            if status == "correct":
                total_correct += 1
            elif status == "substitution" and ref_word and hyp_word:
                total_errors += 1
                failed_words[ref_word] += 1
                substitution_pairs[f"{ref_word} → {hyp_word}"] += 1
            elif status == "deletion" and ref_word:
                total_errors += 1
                failed_words[ref_word] += 1
                deletion_words[ref_word] += 1
            elif status == "insertion":
                total_errors += 1

    # Compute averages
    avg_wer = {}
    for diff, scores in wer_by_difficulty.items():
        if scores:
            avg_wer[diff] = round(sum(scores) / len(scores), 3)

    overall_wer = round(
        total_errors / (total_correct + total_errors), 3
    ) if (total_correct + total_errors) > 0 else 0

    return {
        "has_data": True,
        "overall_wer": overall_wer,
        "wer_by_difficulty": avg_wer,
        "most_failed_words": failed_words.most_common(30),
        "common_substitutions": substitution_pairs.most_common(15),
        "common_deletions": deletion_words.most_common(15),
        "total_transcriptions": len(rows),
    }


def _build_gemini_prompt(round_num: int, performance: dict) -> str:
    """Build the Gemini prompt with performance data for adaptive text generation."""

    base_instructions = """You are a Hebrew speech therapy assistant helping a teenage girl with a speech disability practice reading aloud. Generate exactly 50 Hebrew reading texts as a JSON array.

Rules for ALL texts:
- Write in modern Hebrew, no nikud (ניקוד)
- Natural, conversational tone appropriate for a 17-year-old girl
- Diverse topics: nature, daily life, science, history, technology, sports, cooking, music, animals, space, travel, art, etc.
- Each text object has: {"title": "...", "content": "...", "difficulty": "easy|medium|hard"}
- Easy: 2-3 short sentences, simple common words (15 texts)
- Medium: 3-4 sentences, everyday vocabulary (20 texts)
- Hard: 4-5 sentences, richer vocabulary (15 texts)
- Do NOT use nikud, do NOT repeat topics from previous rounds
- Return ONLY the JSON array, no markdown, no explanation"""

    if not performance.get("has_data"):
        # Round 1 — no prior data
        return f"""{base_instructions}

This is round {round_num + 1}, and we don't have performance data yet. Generate general practice texts with diverse topics and vocabulary."""

    # Adaptive prompt with performance data
    failed_words_str = ", ".join(
        f'"{w}" ({c} errors)' for w, c in performance.get("most_failed_words", [])[:20]
    )
    subs_str = ", ".join(
        f'"{pair}" ({c}x)' for pair, c in performance.get("common_substitutions", [])[:10]
    )
    deletions_str = ", ".join(
        f'"{w}" ({c}x)' for w, c in performance.get("common_deletions", [])[:10]
    )
    wer_by_diff = performance.get("wer_by_difficulty", {})

    # Adjust difficulty distribution based on performance
    easy_wer = wer_by_diff.get("easy", 0)
    medium_wer = wer_by_diff.get("medium", 0)
    hard_wer = wer_by_diff.get("hard", 0)

    difficulty_note = ""
    if easy_wer > 0.4:
        difficulty_note = "The student struggles even with easy texts. Generate MORE easy texts (25 easy, 15 medium, 10 hard)."
    elif hard_wer < 0.15:
        difficulty_note = "The student handles hard texts well. Generate MORE hard texts (10 easy, 15 medium, 25 hard)."

    return f"""{base_instructions}

This is round {round_num + 1}. Here is the student's performance analysis from round {round_num}:

Overall WER: {performance.get('overall_wer', 'N/A')} ({int(performance.get('overall_wer', 0) * 100)}% of words had errors)
WER by difficulty: easy={wer_by_diff.get('easy', 'N/A')}, medium={wer_by_diff.get('medium', 'N/A')}, hard={wer_by_diff.get('hard', 'N/A')}

Words she struggled with most: {failed_words_str or 'None'}
Common substitution patterns (what she said instead): {subs_str or 'None'}
Words she often skipped/deleted: {deletions_str or 'None'}

{difficulty_note}

IMPORTANT adaptation instructions:
1. Naturally weave the struggled words into new text contexts so she practices them again
2. Include words phonetically similar to her common confusions
3. Gradually introduce the struggled words — don't overload a single text
4. Mix targeted practice with fresh vocabulary to keep it engaging
5. If she has specific sound patterns she struggles with (visible in substitutions), include more words with those sounds"""


async def generate_texts_with_gemini(
    round_num: int, performance: dict
) -> list[dict]:
    """Call Gemini to generate 50 adaptive Hebrew texts."""

    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = _build_gemini_prompt(round_num, performance)

    logger.info(f"Calling Gemini ({GEMINI_MODEL}) to generate round {round_num + 1} texts...")

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.9,
            max_output_tokens=16000,
            response_mime_type="application/json",
        ),
    )

    # Parse JSON response
    raw_text = response.text.strip()

    # Handle potential markdown code blocks
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]  # Remove first line
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

    texts = json.loads(raw_text)

    if not isinstance(texts, list):
        raise ValueError("Gemini response is not a JSON array")

    # Validate each text
    valid_texts = []
    for t in texts:
        if not isinstance(t, dict):
            continue
        title = t.get("title", "").strip()
        content = t.get("content", "").strip()
        difficulty = t.get("difficulty", "medium").strip()
        if not title or not content:
            continue
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"
        valid_texts.append({
            "title": title,
            "content": content,
            "difficulty": difficulty,
        })

    if len(valid_texts) < 10:
        raise ValueError(f"Gemini only generated {len(valid_texts)} valid texts (expected ~50)")

    logger.info(f"Gemini generated {len(valid_texts)} texts for round {round_num + 1}")
    return valid_texts


async def create_next_round(db: AsyncSession) -> dict:
    """Analyze current round performance, generate next round via Gemini, insert into DB."""

    # Find current round
    max_round_result = await db.execute(select(func.max(Text.round)))
    current_round = max_round_result.scalar_one() or 1

    # Analyze performance
    performance = await analyze_round_performance(current_round, db)

    # Generate texts
    generated = await generate_texts_with_gemini(current_round, performance)

    new_round = current_round + 1

    # Insert into DB
    created_texts = []
    for t in generated:
        text = Text(
            title=t["title"],
            content=t["content"],
            difficulty=t["difficulty"],
            category="generated",
            word_count=len(t["content"].split()),
            is_builtin=True,
            round=new_round,
        )
        db.add(text)
        created_texts.append(text)

    await db.flush()

    return {
        "round": new_round,
        "texts_created": len(created_texts),
        "performance_analysis": performance,
    }
