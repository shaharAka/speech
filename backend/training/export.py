"""Merge LoRA adapter into base model and convert to CTranslate2."""

import logging
import os
import subprocess

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from training.config import TRAINING_CONFIG

logger = logging.getLogger(__name__)


def merge_and_export(
    adapter_path: str,
    output_dir: str,
    base_model: str | None = None,
) -> str:
    """
    Merge LoRA adapter into base model and convert to CTranslate2 format.

    Args:
        adapter_path: Path to the LoRA adapter directory
        output_dir: Where to save the final CT2 model
        base_model: Base model name (defaults to training config)

    Returns:
        Path to the CT2 model directory
    """
    base_model = base_model or TRAINING_CONFIG["base_model"]

    # Step 1: Load base model and merge LoRA adapter
    logger.info(f"Loading base model: {base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    processor = WhisperProcessor.from_pretrained(base_model)

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    # Step 2: Save merged model
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    logger.info(f"Merged model saved to {merged_dir}")

    # Step 3: Convert to CTranslate2
    ct2_dir = os.path.join(output_dir, "ct2")
    os.makedirs(ct2_dir, exist_ok=True)

    logger.info("Converting to CTranslate2 format...")
    cmd = [
        "ct2-transformers-converter",
        "--model", merged_dir,
        "--output_dir", ct2_dir,
        "--quantization", "float16",
        "--force",
    ]
    # Only copy files that actually exist in the merged dir
    copy_files = [f for f in ["tokenizer.json", "preprocessor_config.json"]
                  if os.path.exists(os.path.join(merged_dir, f))]
    if copy_files:
        cmd.extend(["--copy_files"] + copy_files)
    subprocess.run(cmd, check=True)

    logger.info(f"CT2 model saved to {ct2_dir}")
    return ct2_dir
