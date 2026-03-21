"""LoRA fine-tuning orchestration for Whisper Hebrew."""

import logging
import os

import evaluate
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from app.core.hebrew_utils import normalize_hebrew
from training.config import TRAINING_CONFIG
from training.data_collator import DataCollatorSpeechSeq2SeqWithPadding

logger = logging.getLogger(__name__)


class EpochMetricsCallback(TrainerCallback):
    """Collect loss at the end of each epoch for progress analysis."""

    def __init__(self):
        self.epoch_metrics: list[dict] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.log_history has all logged entries; grab the latest training loss
        epoch = int(state.epoch)
        loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry:
                loss = entry["loss"]
                break
        self.epoch_metrics.append({"epoch": epoch, "loss": round(loss, 4) if loss else None})
        logger.info(f"Epoch {epoch} complete — loss: {loss:.4f}" if loss else f"Epoch {epoch} complete")


def prepare_dataset(batch, processor):
    """Process a single example: extract features and tokenize labels."""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def run_training(
    train_dataset,
    eval_dataset,
    output_dir: str,
    config: dict | None = None,
) -> dict:
    """
    Run LoRA fine-tuning on the Whisper model.

    Args:
        train_dataset: HuggingFace Dataset with 'audio' and 'sentence' columns
        eval_dataset: HuggingFace Dataset for evaluation
        output_dir: Where to save the fine-tuned model
        config: Override training config

    Returns:
        Dict with training metrics
    """
    cfg = {**TRAINING_CONFIG, **(config or {})}

    logger.info(f"Loading base model: {cfg['base_model']}")
    processor = WhisperProcessor.from_pretrained(
        cfg["base_model"], language=cfg["language"], task=cfg["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])

    # Force decoder language settings
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None

    # Build LoRA target modules: decoder attention + encoder layers
    base_modules = list(cfg["lora_target_modules"])  # e.g. ["q_proj", "v_proj", "k_proj", "out_proj"]
    target_modules = list(base_modules)  # decoder (matched by name across all decoder layers)

    # Add explicit encoder layer targeting
    encoder_layers = cfg.get("lora_encoder_layers", [])
    if encoder_layers:
        for layer_idx in encoder_layers:
            for mod in base_modules:
                target_modules.append(f"model.encoder.layers.{layer_idx}.self_attn.{mod}")
        logger.info(f"Encoder LoRA: targeting {len(encoder_layers)} layers ({encoder_layers})")

    # Also add decoder cross-attention (attends to encoder output — critical for speech)
    decoder_layers = cfg.get("lora_decoder_layers", [])
    if decoder_layers:
        for layer_idx in decoder_layers:
            for mod in base_modules:
                target_modules.append(f"model.decoder.layers.{layer_idx}.encoder_attn.{mod}")
        logger.info(f"Decoder cross-attn LoRA: targeting {len(decoder_layers)} layers")

    logger.info(f"Total LoRA targets: {len(target_modules)} modules, rank={cfg['lora_r']}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=target_modules,
        bias=cfg["lora_bias"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tune SpecAugment for flat/monotone speech
    if cfg.get("spec_augment_freq_mask"):
        model.config.mask_feature_length = cfg["spec_augment_freq_mask"]
    if cfg.get("spec_augment_time_mask"):
        model.config.mask_time_length = cfg["spec_augment_time_mask"]

    # Process datasets
    logger.info("Processing datasets...")
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=eval_dataset.column_names,
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # Normalize Hebrew
        pred_str = [normalize_hebrew(p) for p in pred_str]
        label_str = [normalize_hebrew(l) for l in label_str]
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        fp16=cfg["fp16"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["none"],
    )

    epoch_cb = EpochMetricsCallback()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[epoch_cb],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    # Save LoRA adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    logger.info(f"Training complete. Adapter saved to {adapter_dir}")

    # Evaluate
    eval_result = trainer.evaluate()

    return {
        "train_loss": train_result.training_loss,
        "eval_wer": eval_result.get("eval_wer"),
        "adapter_path": adapter_dir,
        "epoch_metrics": epoch_cb.epoch_metrics,
    }
