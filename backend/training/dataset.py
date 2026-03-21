"""Build HuggingFace Dataset from database recordings."""

import os

from datasets import Audio, Dataset
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.models.recording import Recording
from app.models.text import Text


def build_training_dataset(db_session, storage_root: str) -> Dataset:
    """
    Query all recordings that have transcriptions and build a HuggingFace Dataset.

    Each sample has:
      - audio: path to WAV file (will be loaded by HF datasets)
      - sentence: the reference text from the texts table
    """
    results = db_session.execute(
        select(Recording)
        .options(joinedload(Recording.text))
        .where(Recording.transcription != None)  # noqa: E711
    ).scalars().all()

    audio_paths = []
    sentences = []

    for recording in results:
        audio_path = os.path.join(storage_root, recording.audio_path)
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)
            sentences.append(recording.text.content)

    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "sentence": sentences,
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset
