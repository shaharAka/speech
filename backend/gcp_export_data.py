"""Export training data from local SQLite DB for GCP training.

Creates a directory with:
  - manifest.json: metadata + sample list
  - *.wav: audio files (copied flat)
"""

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path


def export_training_data(db_path: str, audio_root: str, output_dir: str, config: dict):
    """Export recordings with transcriptions for remote training."""
    os.makedirs(output_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get recordings that have transcriptions (i.e., usable for training)
    query = """
        SELECT r.id, r.audio_path, t.content as sentence
        FROM recordings r
        JOIN texts t ON r.text_id = t.id
        JOIN transcriptions tr ON tr.recording_id = r.id
        WHERE r.audio_path IS NOT NULL
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    samples = []
    for row in rows:
        src_audio = os.path.join(audio_root, row["audio_path"])
        if not os.path.exists(src_audio):
            print(f"  WARNING: Audio file not found: {src_audio}")
            continue

        # Copy audio file with flat name
        audio_filename = f"sample_{row['id']}.wav"
        dst_audio = os.path.join(output_dir, audio_filename)
        shutil.copy2(src_audio, dst_audio)

        samples.append({
            "id": row["id"],
            "audio_file": audio_filename,
            "sentence": row["sentence"],
        })

    # Write manifest
    manifest = {
        "config": config,
        "num_samples": len(samples),
        "samples": samples,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(samples)} samples to {output_dir}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training data for GCP")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--audio-root", required=True, help="Root directory for audio files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", default="{}", help="Training config JSON string")
    args = parser.parse_args()

    config = json.loads(args.config)
    export_training_data(args.db, args.audio_root, args.output_dir, config)
