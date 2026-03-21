"""Vertex AI training entrypoint — runs inside the pre-built PyTorch container.

Downloads training data from GCS, runs LoRA fine-tuning, exports CT2 model,
and uploads results back to GCS.
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vertex_entrypoint")


def _run(cmd: list[str], **kwargs):
    logger.info(f"Running: {' '.join(cmd[:6])}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-bucket", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    args = parser.parse_args()

    bucket = args.gcs_bucket
    run_id = args.run_id
    job_dir = f"gs://{bucket}/training-jobs/run_{run_id}"

    # ── 1. Install project code from GCS ────────────────────────────
    logger.info("Downloading code from GCS...")
    os.makedirs("/opt/whisper", exist_ok=True)
    _run(["gsutil", "-m", "cp", "-r", f"gs://{bucket}/code/*", "/opt/whisper/"])
    os.chdir("/opt/whisper")

    # Install the project package (includes training extras)
    _run([sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[training]",
          "ctranslate2", "aiosqlite"])

    # ── 2. Download training data ───────────────────────────────────
    logger.info("Downloading training data...")
    os.makedirs("/tmp/training-data", exist_ok=True)
    _run(["gsutil", "-m", "cp", "-r", f"{job_dir}/data/*", "/tmp/training-data/"])
    _run(["gsutil", "cp", f"{job_dir}/manifest.json", "/tmp/training-data/manifest.json"])

    # ── 3. Load manifest ────────────────────────────────────────────
    with open("/tmp/training-data/manifest.json") as f:
        manifest = json.load(f)

    config = manifest["config"]
    samples = manifest["samples"]
    logger.info(f"Manifest: {len(samples)} samples, config={config}")

    # ── 4. Build dataset ────────────────────────────────────────────
    from datasets import Audio, Dataset

    audio_paths = [os.path.join("/tmp/training-data", s["audio_file"]) for s in samples]
    sentences = [s["sentence"] for s in samples]
    existing = [(a, s) for a, s in zip(audio_paths, sentences) if os.path.exists(a)]
    logger.info(f"Valid audio files: {len(existing)}/{len(audio_paths)}")

    if len(existing) < 10:
        logger.error("Not enough valid samples (need >= 10)")
        sys.exit(1)

    audio_paths, sentences = zip(*existing)
    dataset = Dataset.from_dict({"audio": list(audio_paths), "sentence": list(sentences)})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # ── 5. Run training ─────────────────────────────────────────────
    logger.info("Starting LoRA fine-tuning...")
    from training.trainer import run_training

    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)

    result = run_training(
        train_dataset=split["train"],
        eval_dataset=split["test"],
        output_dir=output_dir,
        config={
            "num_train_epochs": config.get("num_epochs", 5),
            "lora_r": config.get("lora_rank", 32),
            "learning_rate": config.get("learning_rate", 1e-4),
        },
    )

    # ── 6. Export to CT2 ─────────────────────────────────────────────
    logger.info("Merging LoRA adapter and converting to CT2...")
    from training.export import merge_and_export

    ct2_path = merge_and_export(result["adapter_path"], os.path.join(output_dir, "final"))

    # ── 7. Write results ─────────────────────────────────────────────
    results = {
        "status": "completed",
        "train_loss": result.get("train_loss"),
        "eval_wer": result.get("eval_wer"),
        "ct2_path": ct2_path,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training complete: {results}")

    # ── 8. Upload results to GCS ─────────────────────────────────────
    logger.info("Uploading results to GCS...")
    _run(["gsutil", "-m", "cp", "-r", f"{output_dir}/final/", f"{job_dir}/model/"])
    _run(["gsutil", "cp", f"{output_dir}/results.json", f"{job_dir}/results.json"])
    if os.path.isdir(os.path.join(output_dir, "adapter")):
        _run(["gsutil", "-m", "cp", "-r", f"{output_dir}/adapter/", f"{job_dir}/adapter/"])

    logger.info("All done — results uploaded to GCS.")


if __name__ == "__main__":
    main()
