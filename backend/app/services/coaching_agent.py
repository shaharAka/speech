"""Speech Therapy Coaching Agent — Gemini analyzes training results and generates adaptive practice texts.

This module runs SYNCHRONOUSLY inside Celery tasks (not async).
"""

import json
import logging
from collections import Counter

import google.generativeai as genai
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.config import settings
from app.models.coaching_report import CoachingReport
from app.models.recording import Recording
from app.models.text import Text
from app.models.training_run import TrainingRun
from app.models.transcription import Transcription

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


def analyze_all_rounds_sync(db: Session) -> dict:
    """Analyze transcription data across ALL rounds to build comprehensive error profile."""

    # WER trajectory from training runs
    runs = db.execute(
        select(TrainingRun)
        .where(TrainingRun.status == "completed")
        .order_by(TrainingRun.created_at.asc())
    ).scalars().all()

    wer_trajectory = []
    for r in runs:
        wer_trajectory.append({
            "run_id": r.id,
            "eval_wer": r.eval_wer,
            "train_wer": r.train_wer,
            "loss": r.training_loss,
            "num_samples": r.num_samples,
            "epochs": r.num_epochs,
        })

    # Find current round
    max_round = db.execute(select(func.max(Text.round))).scalar_one() or 1

    # Get all text IDs grouped by round
    all_texts = db.execute(
        select(Text.id, Text.difficulty, Text.round)
    ).all()
    text_difficulty = {r[0]: r[1] for r in all_texts}
    text_round = {r[0]: r[2] for r in all_texts}

    # Get all transcriptions
    transcriptions = db.execute(
        select(Transcription, Recording.text_id)
        .join(Recording, Transcription.recording_id == Recording.id)
    ).all()

    if not transcriptions:
        return {
            "has_data": False,
            "wer_trajectory": wer_trajectory,
            "round_count": max_round,
            "total_recordings": 0,
        }

    # Per-round error aggregation
    per_round_errors: dict[int, Counter] = {}
    wer_by_difficulty: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    failed_words: Counter = Counter()
    substitution_pairs: Counter = Counter()
    deletion_words: Counter = Counter()
    total_correct = 0
    total_errors = 0

    for transcription, text_id in transcriptions:
        difficulty = text_difficulty.get(text_id, "medium")
        round_num = text_round.get(text_id, 1)
        wer_by_difficulty[difficulty].append(transcription.wer_score)

        if round_num not in per_round_errors:
            per_round_errors[round_num] = Counter()

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
                per_round_errors[round_num][ref_word] += 1
                substitution_pairs[f"{ref_word} → {hyp_word}"] += 1
            elif status == "deletion" and ref_word:
                total_errors += 1
                failed_words[ref_word] += 1
                per_round_errors[round_num][ref_word] += 1
                deletion_words[ref_word] += 1
            elif status == "insertion":
                total_errors += 1

    # Identify persistent errors (failed in 2+ rounds)
    word_round_counts: Counter = Counter()
    for round_errors in per_round_errors.values():
        for word in round_errors:
            word_round_counts[word] += 1
    persistent_errors = [
        (word, count) for word, count in word_round_counts.most_common(30)
        if count >= 2
    ]

    # Average WER by difficulty
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
        "persistent_errors": persistent_errors,
        "wer_trajectory": wer_trajectory,
        "round_count": max_round,
        "total_recordings": len(transcriptions),
    }


def _build_coaching_prompt(analysis: dict, training_run: TrainingRun) -> str:
    """Build the enhanced Gemini prompt for the coaching agent."""

    is_round1 = analysis.get("round_count", 1) <= 1

    # WER trajectory table
    trajectory = analysis.get("wer_trajectory", [])
    trajectory_str = "No prior training runs."
    if trajectory:
        lines = ["Run | Eval WER | Train WER | Loss | Samples | Epochs"]
        for t in trajectory:
            eval_wer = f"{t['eval_wer']:.3f}" if t['eval_wer'] is not None else 'N/A'
            train_wer = f"{t['train_wer']:.3f}" if t['train_wer'] is not None else 'N/A'
            loss = f"{t['loss']:.4f}" if t['loss'] is not None else 'N/A'
            lines.append(
                f"#{t['run_id']} | {eval_wer} | {train_wer} | {loss} | "
                f"{t['num_samples']} | {t['epochs']}"
            )
        trajectory_str = "\n".join(lines)

    # Current run metrics
    current_run_str = (
        f"Run #{training_run.id}: {training_run.num_epochs} epochs, "
        f"lr={training_run.learning_rate}, lora_rank={training_run.lora_rank}\n"
        f"Eval WER: {training_run.eval_wer or 'N/A'} | "
        f"Train WER: {training_run.train_wer or 'N/A'} | "
        f"Loss: {training_run.training_loss or 'N/A'}"
    )

    # Error patterns
    failed_words_str = ", ".join(
        f'"{w}" ({c}x)' for w, c in analysis.get("most_failed_words", [])[:20]
    ) or "None"
    subs_str = ", ".join(
        f'"{p}" ({c}x)' for p, c in analysis.get("common_substitutions", [])[:10]
    ) or "None"
    deletions_str = ", ".join(
        f'"{w}" ({c}x)' for w, c in analysis.get("common_deletions", [])[:10]
    ) or "None"
    persistent_str = ", ".join(
        f'"{w}" ({c} rounds)' for w, c in analysis.get("persistent_errors", [])[:15]
    ) or "None"

    wer_by_diff = analysis.get("wer_by_difficulty", {})

    round1_disclaimer = ""
    if is_round1:
        round1_disclaimer = """
IMPORTANT: This is the first training run (round 1 data). The base model cannot understand
this student's speech at all, so transcription error data is mostly NOISE. Do NOT draw
conclusions from specific word errors. Generate diverse general-purpose texts. Your insights
should note that this is the baseline and meaningful analysis begins from round 2."""

    return f"""You are a speech therapy AI agent coaching a 17-year-old girl with a speech disability
who is practicing Hebrew reading aloud. A Whisper speech recognition model has just been
fine-tuned on her recordings.

Your job:
1. Analyze the training results and transcription error patterns
2. Provide coaching insights in Hebrew
3. Generate the next 50 practice texts optimized for her improvement

=== TRAINING HISTORY (WER TRAJECTORY) ===
{trajectory_str}

=== CURRENT TRAINING RUN ===
{current_run_str}

=== ERROR ANALYSIS (CUMULATIVE ACROSS ALL ROUNDS) ===
Overall WER: {analysis.get('overall_wer', 'N/A')}
WER by difficulty: easy={wer_by_diff.get('easy', 'N/A')}, medium={wer_by_diff.get('medium', 'N/A')}, hard={wer_by_diff.get('hard', 'N/A')}
Most failed words: {failed_words_str}
Common substitutions: {subs_str}
Common deletions: {deletions_str}
Total recordings analyzed: {analysis.get('total_recordings', 0)}

=== PERSISTENT ERRORS (failed in 2+ rounds) ===
{persistent_str}
{round1_disclaimer}

Respond with a single JSON object (no markdown, no explanation) with these keys:

{{
  "summary": "Hebrew-language coaching summary for the student (3-5 sentences, encouraging tone, specific to her progress)",
  "insights": [
    {{"category": "improvement|regression|pattern|milestone", "message_he": "Hebrew insight text", "severity": "info|warning|success"}}
  ],
  "recommendations": [
    {{"type": "focus_area|difficulty|training_params|text_length|general", "detail": "Specific recommendation in Hebrew", "priority": "high|medium|low"}}
  ],
  "difficulty_distribution": {{"easy": N, "medium": N, "hard": N}},
  "suggested_params": {{"num_epochs": N, "learning_rate": N}} or null,
  "texts": [
    {{"title": "Hebrew title", "content": "Hebrew text content (no nikud)", "difficulty": "easy|medium|hard"}}
  ]
}}

Rules for texts:
- Generate exactly 50 texts
- Modern Hebrew, no nikud, conversational tone for a 17-year-old
- IMPORTANT: Choose FUN, INTERESTING topics that a 17-year-old girl would enjoy reading!
  Pick a specific engaging theme per text (not generic). Examples of good topics:
  social media & influencers, K-pop & music artists, fashion & makeup trends,
  Netflix/movies/TV shows, cooking & baking recipes, space & astronomy facts,
  ocean creatures, psychology & personality types, travel destinations,
  photography tips, DIY crafts, gaming culture, viral TikTok trends,
  true crime stories, fun science experiments, animal rescue stories,
  music production, sports highlights, funny cultural facts, AI art.
  Each text should feel like a fun snippet she'd WANT to read, not a textbook exercise.
- Easy: 2-3 short sentences | Medium: 3-4 sentences | Hard: 4-5 sentences
- Naturally weave struggled words into new contexts
- Include phonetically similar words to common confusions
- Adjust difficulty distribution based on the student's WER per difficulty level
- difficulty_distribution values MUST sum to 50"""


def generate_coaching_report_sync(training_run_id: int, db: Session) -> CoachingReport:
    """Main orchestration: analyze → prompt Gemini → insert texts → create report."""

    # Load training run
    run = db.execute(
        select(TrainingRun).where(TrainingRun.id == training_run_id)
    ).scalar_one()

    # Current round
    max_round = db.execute(select(func.max(Text.round))).scalar_one() or 1
    is_round1 = max_round <= 1

    # Analyze all data
    logger.info(f"[Coaching] Analyzing all rounds for run #{training_run_id}...")
    analysis = analyze_all_rounds_sync(db)

    # Build prompt and call Gemini
    logger.info(f"[Coaching] Calling Gemini for coaching report...")
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = _build_coaching_prompt(analysis, run)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.8,
            max_output_tokens=20000,
            response_mime_type="application/json",
        ),
    )

    # Parse response
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

    result = json.loads(raw_text)
    if not isinstance(result, dict):
        raise ValueError("Gemini response is not a JSON object")

    # Extract and validate texts
    texts = result.get("texts", [])
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
        valid_texts.append({"title": title, "content": content, "difficulty": difficulty})

    if len(valid_texts) < 10:
        raise ValueError(f"Gemini generated only {len(valid_texts)} valid texts (need >= 10)")

    logger.info(f"[Coaching] Gemini generated {len(valid_texts)} texts")

    # Insert texts for next round
    new_round = max_round + 1
    for t in valid_texts:
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
    db.flush()

    logger.info(f"[Coaching] Inserted {len(valid_texts)} texts for round {new_round}")

    # Build coaching report
    difficulty_dist = result.get("difficulty_distribution", {"easy": 15, "medium": 20, "hard": 15})
    report = CoachingReport(
        training_run_id=training_run_id,
        round_number=max_round,
        next_round_number=new_round,
        summary_text=result.get("summary", ""),
        insights_json=json.dumps(result.get("insights", []), ensure_ascii=False),
        recommendations_json=json.dumps(result.get("recommendations", []), ensure_ascii=False),
        wer_trajectory_json=json.dumps(analysis.get("wer_trajectory", []), ensure_ascii=False),
        error_analysis_json=json.dumps({
            "overall_wer": analysis.get("overall_wer"),
            "wer_by_difficulty": analysis.get("wer_by_difficulty", {}),
            "most_failed_words": analysis.get("most_failed_words", [])[:20],
            "persistent_errors": analysis.get("persistent_errors", [])[:10],
        }, ensure_ascii=False),
        difficulty_distribution_json=json.dumps(difficulty_dist, ensure_ascii=False),
        suggested_next_params_json=json.dumps(
            result.get("suggested_params"), ensure_ascii=False
        ) if result.get("suggested_params") else None,
        texts_generated=len(valid_texts),
        is_round1_noise=is_round1,
    )
    db.add(report)
    db.flush()

    logger.info(f"[Coaching] Report #{report.id} created for run #{training_run_id}")
    return report
