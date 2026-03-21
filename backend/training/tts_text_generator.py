"""Generate diverse Hebrew sentences for TTS data augmentation via Gemini."""

import json
import logging

import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


def generate_tts_texts(
    num_texts: int = 500,
    gemini_api_key: str = "",
) -> list[str]:
    """Generate diverse Hebrew sentences for TTS synthesis.

    Produces short-to-medium sentences (5-15 words) optimized for TTS quality.
    Shorter sentences produce cleaner TTS output.
    """
    if not gemini_api_key:
        logger.warning("No Gemini API key — returning empty TTS texts")
        return []

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Generate in batches to avoid response size limits
    batch_size = 100
    all_texts: list[str] = []

    for batch_idx in range(0, num_texts, batch_size):
        remaining = min(batch_size, num_texts - len(all_texts))

        prompt = f"""Generate exactly {remaining} diverse Hebrew sentences for text-to-speech synthesis.

Rules:
- Each sentence must be 5-15 words long (shorter is better for TTS quality)
- Modern Hebrew, no nikud, natural conversational tone
- DIVERSE topics: daily life, nature, food, technology, school, music, sports,
  animals, weather, travel, emotions, family, hobbies, science, history
- Mix of statement types: descriptions, opinions, questions, exclamations
- Include common Hebrew words and natural sentence structures
- NO titles, NO numbering — just plain sentences
- Each sentence on its own line

Respond with ONLY the sentences, one per line, nothing else."""

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=1.0,
                    max_output_tokens=8192,
                ),
            )
            lines = [
                line.strip()
                for line in response.text.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            all_texts.extend(lines[:remaining])
            logger.info(f"TTS text batch {batch_idx // batch_size + 1}: generated {len(lines)} sentences")
        except Exception as e:
            logger.error(f"Gemini TTS text generation failed: {e}")
            break

    logger.info(f"Total TTS texts generated: {len(all_texts)}")
    return all_texts[:num_texts]
