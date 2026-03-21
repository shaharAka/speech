"""LoRA fine-tuning hyperparameters for Whisper Hebrew adaptation."""

TRAINING_CONFIG = {
    # Base model (HuggingFace Transformers format for training)
    "base_model": "ivrit-ai/whisper-large-v3-turbo",
    "language": "he",
    "task": "transcribe",

    # LoRA configuration — aggressive for atypical speech adaptation
    "lora_r": 64,         # doubled from 32 for more capacity
    "lora_alpha": 128,    # 2x rank
    "lora_dropout": 0.1,  # slightly more regularization with deeper training
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],  # all attention projections
    "lora_bias": "none",
    # Encoder LoRA — broad range for flat/monotone speech
    # Low layers (0-5): acoustic features, pitch, formants
    # Mid layers (6-15): phoneme/word boundaries
    # High layers (16+): language modeling
    "lora_encoder_layers": list(range(0, 20, 2)),  # every other layer: 0,2,4,...,18

    # SpecAugment tuning for flat/monotone speech
    "spec_augment_freq_mask": 15,    # reduced from default 27 — less freq masking
    "spec_augment_time_mask": 100,   # increased — more time masking for monotone speech

    # TTS data augmentation (pre-built Docker image has ChatterboxTTS installed)
    "tts_enabled": True,
    "tts_num_synthetic": 500,
    "tts_reference_clips": 5,
    "real_sample_weight": 8,  # oversample real data to balance with synthetic

    # Training arguments
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-5,    # lower LR for deeper training — less risk of catastrophic forgetting
    "warmup_steps": 100,      # longer warmup for more parameters
    "num_train_epochs": 5,
    "fp16": True,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "logging_steps": 25,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,

    # Data split
    "eval_split_ratio": 0.1,
}
