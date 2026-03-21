"""Data collator for Whisper Seq2Seq training."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input features and labels for Whisper training.
    Input features are padded to the maximum length in the batch.
    Labels are padded with -100 so they are ignored in loss computation.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Split inputs and labels
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the BOS token if it was prepended during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
