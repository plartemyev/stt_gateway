#!/usr/bin/env python
"""Prepare FLEURS benchmark data: 16 kHz mono s16 WAVs + reference transcripts.

Google FLEURS covers 102 languages; we use en_us, ru_ru and hy_am — the three
languages the translation pipeline must support.

Usage:
    python benchmarks/prepare_data.py --out benchmarks/data --samples 100
"""

import argparse
import io
import json
import pathlib
import sys
import wave

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

# FLEURS config -> whisper language code. Revision pinned for reproducible
# benchmarks (S615).
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_LANGUAGES = {
    "en_us": "en",
    "ru_ru": "ru",
    "hy_am": "hy",
}
TARGET_RATE = 16_000


def decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Decode a HF datasets audio record (raw bytes) to float32 samples."""
    samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def write_wav(path: pathlib.Path, array: np.ndarray, sample_rate: int) -> None:
    """Write mono float32/int16 samples as a 16 kHz s16 WAV file."""
    if sample_rate != TARGET_RATE:
        raise SystemExit(f"FLEURS audio at {sample_rate} Hz, expected {TARGET_RATE}")
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.dtype != np.float32:
        array = array.astype(np.float32) / 32768.0
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_RATE)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/data", help="output directory")
    parser.add_argument("--samples", type=int, default=100, help="samples per language")
    parser.add_argument("--split", default="test", choices=("test", "train", "validation"))
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fleurs_config, language in FLEURS_LANGUAGES.items():
        dataset = load_dataset("google/fleurs", fleurs_config, split=args.split, revision=FLEURS_REVISION)
        # Raw bytes only: column access must not trigger the built-in audio
        # decoder (torchcodec); we decode with soundfile ourselves.
        dataset = dataset.cast_column("audio", Audio(decode=False))
        manifest_path = out_dir / f"{fleurs_config}.jsonl"
        written = 0
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in dataset:
                if written >= args.samples:
                    break
                transcript = " ".join(str(row["transcription"]).split())
                if not transcript:
                    continue
                audio_path = out_dir / f"{fleurs_config}_{written:04d}.wav"
                array, sample_rate = decode_audio(row["audio"])
                write_wav(audio_path, array, sample_rate)
                manifest.write(json.dumps({
                    "file": audio_path.name,
                    "language": language,
                    "transcript": transcript,
                }) + "\n")
                written += 1
        print(f"{fleurs_config}: wrote {written} samples -> {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
