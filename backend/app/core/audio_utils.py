import json
import os
import subprocess
import tempfile


def convert_to_wav_16k(input_path: str, output_path: str) -> None:
    """Convert any audio file to WAV 16kHz mono (what Whisper expects)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            output_path,
        ],
        capture_output=True,
        check=True,
    )


def get_audio_duration_ms(file_path: str) -> int:
    """Get audio duration in milliseconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", file_path,
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    duration_sec = float(data["format"]["duration"])
    return int(duration_sec * 1000)


async def save_upload_to_temp(content: bytes, suffix: str = ".webm") -> str:
    """Save uploaded bytes to a temporary file, return path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path
