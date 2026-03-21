"""TTS data augmentation using Chatterbox voice cloning for dysarthric speech training."""

import logging
import os

import torch
import torchaudio

logger = logging.getLogger(__name__)


def select_reference_clips(manifest: dict, data_dir: str, num_clips: int = 5) -> list[str]:
    """Pick the N shortest real recordings as TTS reference clips.

    Shorter clips tend to be cleaner and work better for voice cloning.
    """
    samples = [s for s in manifest["samples"] if s.get("source", "real") == "real"]
    # Sort by file size (proxy for duration) — smaller = shorter
    sized = []
    for s in samples:
        path = os.path.join(data_dir, s["audio_file"])
        if os.path.exists(path):
            sized.append((os.path.getsize(path), path))
    sized.sort()
    clips = [path for _, path in sized[:num_clips]]
    logger.info(f"Selected {len(clips)} reference clips for voice cloning")
    return clips


def generate_synthetic_audio(
    texts: list[str],
    reference_clips: list[str],
    output_dir: str,
    device: str = "cuda",
) -> list[dict]:
    """Generate synthetic WAV files using Chatterbox TTS with voice cloning.

    Args:
        texts: Hebrew sentences to synthesize
        reference_clips: WAV files of the target speaker's voice
        output_dir: Where to save synthetic WAVs
        device: "cuda" or "cpu"

    Returns:
        List of {"audio_file": "synth_001.wav", "sentence": "...", "source": "synthetic"}
    """
    from chatterbox.tts import ChatterboxTTS

    logger.info(f"Loading Chatterbox TTS on {device}...")
    model = ChatterboxTTS.from_pretrained(device=device)

    os.makedirs(output_dir, exist_ok=True)
    results = []
    num_refs = len(reference_clips)

    for i, text in enumerate(texts):
        # Cycle through reference clips
        ref_clip = reference_clips[i % num_refs]
        synth_filename = f"synth_{i+1:04d}.wav"
        synth_path = os.path.join(output_dir, synth_filename)

        try:
            wav = model.generate(text, audio_prompt_path=ref_clip)
            # Chatterbox outputs at 24kHz — resample to 16kHz for Whisper
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav_16k = torchaudio.functional.resample(wav, orig_freq=24000, new_freq=16000)
            torchaudio.save(synth_path, wav_16k.cpu(), 16000)

            results.append({
                "audio_file": synth_filename,
                "sentence": text,
                "source": "synthetic",
            })

            if (i + 1) % 50 == 0:
                logger.info(f"TTS progress: {i+1}/{len(texts)} samples generated")

        except Exception as e:
            logger.warning(f"TTS failed for sample {i+1}: {e}")
            continue

    logger.info(f"TTS generation complete: {len(results)}/{len(texts)} samples")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def run_tts_augmentation(
    manifest: dict,
    data_dir: str,
    tts_texts: list[str],
    config: dict | None = None,
) -> dict:
    """Main entry: generate synthetic data and return augmented manifest.

    Args:
        manifest: Original manifest with real samples
        data_dir: Directory containing real audio files
        tts_texts: Hebrew sentences to synthesize
        config: Optional config overrides

    Returns:
        Updated manifest with both real and synthetic samples
    """
    cfg = config or {}
    num_refs = cfg.get("tts_reference_clips", 5)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tag existing samples as "real"
    for s in manifest["samples"]:
        s.setdefault("source", "real")

    # Select reference clips
    ref_clips = select_reference_clips(manifest, data_dir, num_clips=num_refs)
    if not ref_clips:
        logger.error("No reference clips found — skipping TTS augmentation")
        return manifest

    # Generate synthetic audio
    synth_dir = os.path.join(data_dir, "synthetic")
    synth_samples = generate_synthetic_audio(
        texts=tts_texts,
        reference_clips=ref_clips,
        output_dir=synth_dir,
        device=device,
    )

    # Update audio_file paths to include synthetic/ prefix
    for s in synth_samples:
        s["audio_file"] = f"synthetic/{s['audio_file']}"

    # Combine
    manifest["samples"].extend(synth_samples)
    manifest["num_synthetic"] = len(synth_samples)
    logger.info(
        f"Augmented manifest: {manifest['num_samples']} real + "
        f"{len(synth_samples)} synthetic = {len(manifest['samples'])} total"
    )

    return manifest
