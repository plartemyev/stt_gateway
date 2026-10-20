#!/usr/bin/env python
r"""WER + RTF benchmark of STT models through comparable GPU paths.

- `ggml-*` models run through the production transcription path: a warm
  whisper-server spawned per model, queried with the gateway's WhisperEngine.
- `seamless-m4t-v2-large` runs in-process via HuggingFace transformers
  (SeamlessM4Tv2ForSpeechToText, greedy decoding) as an open-weights
  reference. Google Chirp 2/3 are cloud-only (GCP STT API v2) and need
  credentials; they are handled by run_bench_gcp.py when supplied.

WER is computed with jiwer on normalized text (lowercase, punctuation
stripped). RTF = transcription wall time / audio duration (<1 = faster than
real time); model load time is reported separately.

Usage:
    python benchmarks/run_bench.py --data benchmarks/data \\
        --models ggml-large-v3-turbo ggml-large-v3 seamless-m4t-v2-large \\
        --models-dir /models --samples 100 --out benchmarks/results.json
"""

import argparse
import asyncio
import contextlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time
import wave

import jiwer
import numpy as np

from stt_gateway.engine import WhisperEngine

# Pinned revision of the Seamless weights used by the benchmark (S615);
# CC-BY-NC — see README "License & model weights".
SEAMLESS_HF_ID = "facebook/seamless-m4t-v2-large"
SEAMLESS_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"

PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")
WHISPER_LANGUAGES = {"en": "en", "ru": "ru", "hy": "hy"}
SEAMLESS_LANGUAGES = {"en": "eng", "ru": "rus", "hy": "hye"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for WER."""
    lowered = PUNCTUATION.sub(" ", text.lower())
    return WHITESPACE.sub(" ", lowered).strip()


def wer(reference: str, hypothesis: str) -> float:
    reference_norm = normalize(reference)
    hypothesis_norm = normalize(hypothesis)
    if not reference_norm:
        return 0.0 if not hypothesis_norm else 1.0
    return jiwer.wer(reference_norm, hypothesis_norm)


def percentile(values: list, percent: int) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * percent / 100), len(ordered) - 1)
    return ordered[index]


def wav_duration_seconds(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def read_wav_16k(path: pathlib.Path) -> "np.ndarray":
    """Read a 16 kHz mono s16 WAV as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != 16_000 or wav_file.getnchannels() != 1:
            raise SystemExit(f"{path}: expected 16 kHz mono WAV")
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return pcm.astype("float32") / 32768.0


class WhisperServerRunner:
    """Production path: warm whisper-server subprocess + WhisperEngine."""

    def __init__(self, model: str, args: argparse.Namespace):
        self.model = model
        self.args = args
        self.process = None
        self.engine = None
        self.load_seconds = 0.0

    async def __aenter__(self) -> "WhisperServerRunner":
        self.process = spawn_engine(self.args.models_dir, self.model, self.args.port, not self.args.no_vad)
        self.engine = WhisperEngine(f"http://127.0.0.1:{self.args.port}", timeout_seconds=300.0)
        started = time.monotonic()
        await self.engine.wait_ready(self.args.engine_start_timeout)
        self.load_seconds = time.monotonic() - started
        return self

    async def transcribe(self, path: pathlib.Path, language: str):
        started = time.monotonic()
        transcription = await self.engine.transcribe(path.read_bytes(), WHISPER_LANGUAGES[language])
        return transcription.text, time.monotonic() - started

    async def __aexit__(self, *exc_info) -> None:
        await self.engine.aclose()
        self.process.terminate()
        self.process.wait(timeout=60)


class SeamlessRunner:
    """Open-weights reference: SeamlessM4T v2 large in-process (transformers)."""

    HF_ID = SEAMLESS_HF_ID
    MAX_NEW_TOKENS = 384

    def __init__(self, model: str, args: argparse.Namespace):
        if model != "seamless-m4t-v2-large":
            raise SystemExit(f"unsupported seamless model: {model}")
        self.args = args
        self.processor = None
        self.model = None
        self.load_seconds = 0.0

    async def __aenter__(self) -> "SeamlessRunner":
        started = time.monotonic()
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.processor = AutoProcessor.from_pretrained(self.HF_ID, revision=SEAMLESS_REVISION)
        self.model = (
            SeamlessM4Tv2ForSpeechToText.from_pretrained(self.HF_ID, revision=SEAMLESS_REVISION)
            .to("cuda")
            .eval()
        )
        with contextlib.suppress(Exception):
            self._generate(read_wav_16k(self.args.data / "en_us_0000.wav"), "en")  # cudnn warmup
        self.load_seconds = time.monotonic() - started
        return self

    def _generate(self, samples, language: str) -> str:
        import torch

        inputs = self.processor(audio=samples, sampling_rate=16_000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # transformers 5.x: the dedicated ForSpeechToText class implies
            # the transcribe task; only tgt_lang is honored here.
            token_ids = self.model.generate(
                **inputs,
                tgt_lang=SEAMLESS_LANGUAGES[language],
                num_beams=1,
                do_sample=False,
                max_new_tokens=self.MAX_NEW_TOKENS,
            )
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0]

    async def transcribe(self, path: pathlib.Path, language: str):
        started = time.monotonic()
        text = self._generate(read_wav_16k(path), language)
        return text.strip(), time.monotonic() - started

    async def __aexit__(self, *exc_info) -> None:
        import torch

        del self.model
        self.model = None
        torch.cuda.empty_cache()


def spawn_engine(models_dir: str, model: str, port: int, vad: bool) -> subprocess.Popen:
    models = pathlib.Path(models_dir)
    command = [
        "/opt/whisper/bin/whisper-server",
        "--model", str(models / f"{model}.bin"),
        "--threads", "8",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--inference-path", "/inference",
    ]
    if vad:
        command += ["--vad", "--vad-model", str(models / "ggml-silero-v6.2.0.bin")]
    print(f"starting: {' '.join(command)}", file=sys.stderr)
    return subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)


def make_runner(model: str, args: argparse.Namespace):
    """Pick the runner by model name: ggml-* -> whisper-server, seamless* -> transformers."""
    if model.startswith("ggml-"):
        return WhisperServerRunner(model, args)
    if model.startswith("seamless"):
        return SeamlessRunner(model, args)
    raise SystemExit(f"unknown model: {model}")


async def benchmark_model(model: str, samples: list, args: argparse.Namespace) -> dict:
    async with make_runner(model, args) as runner:
        rows = []
        for sample in samples:
            audio_path = args.data / sample["file"]
            duration = wav_duration_seconds(audio_path)
            hypothesis, elapsed = await runner.transcribe(audio_path, sample["language"])
            rows.append({
                "file": sample["file"],
                "language": sample["language"],
                "reference": sample["transcript"],
                "hypothesis": hypothesis,
                "duration": round(duration, 3),
                "transcribe_seconds": round(elapsed, 3),
                "rtf": round(elapsed / duration, 4) if duration > 0 else 0.0,
            })
            print(f"  {sample['file']}: rtf={rows[-1]['rtf']:.3f}", file=sys.stderr)
        return {"model": model, "load_seconds": round(runner.load_seconds, 1), "rows": rows}


def flatten(samples_by_language: dict) -> list:
    samples = []
    for language in sorted(samples_by_language):
        samples.extend(samples_by_language[language])
    return samples


def render_report(results: list) -> str:
    lines = [
        "# stt_gateway model benchmark",
        "",
        "FLEURS test split. ggml-* models go through the production transcription path",
        "(gateway WhisperEngine -> warm whisper-server HTTP, Silero VAD on);",
        "seamless-m4t-v2-large runs in-process via transformers, greedy decoding,",
        "no punctuation model (WER normalization strips punctuation anyway).",
        "WER computed on normalized text (lowercase, punctuation stripped).",
        "RTF = transcription wall time / audio duration (<1 means faster than real time).",
        "",
    ]
    for result in results:
        if result.get("error"):
            lines.append(f"## {result['model']} — FAILED")
            lines.append("")
            lines.append(f"```{result['error']}```")
            lines.append("")
            continue
        lines.append(f"## {result['model']}")
        lines.append("")
        lines.append(f"Model load time: {result['load_seconds']}s")
        lines.append("")
        lines.append("| language | samples | WER % | RTF mean | RTF p95 |")
        lines.append("|---|---|---|---|---|")
        for language in sorted({row["language"] for row in result["rows"]}):
            rows = [row for row in result["rows"] if row["language"] == language]
            wers = [wer(row["reference"], row["hypothesis"]) for row in rows]
            rtfs = [row["rtf"] for row in rows]
            lines.append(
                f"| {language} | {len(rows)} | {100 * statistics.fmean(wers):.2f} "
                f"| {statistics.fmean(rtfs):.3f} | {percentile(rtfs, 95):.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("benchmarks/data"))
    parser.add_argument("--models-dir", default="/models")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=100, help="max samples per language")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--engine-start-timeout", type=float, default=900.0)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("benchmarks/results.json"))
    args = parser.parse_args()

    samples_by_language: dict = {}
    for manifest in sorted(args.data.glob("*.jsonl")):
        count = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            if count >= args.samples:
                break
            samples_by_language.setdefault(sample["language"], []).append(sample)
            count += 1

    results = []
    for model in args.models:
        print(f"=== {model} ===", file=sys.stderr)
        try:
            results.append(await benchmark_model(model, flatten(samples_by_language), args))
        except Exception as e:  # noqa: BLE001 - one failed model must not lose the others' results
            print(f"MODEL FAILED: {model}: {e}", file=sys.stderr)
            results.append({"model": model, "load_seconds": 0, "rows": [], "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.out.with_name("report.md")
    report_path.write_text(render_report(results), encoding="utf-8")
    print(render_report(results), file=sys.stderr)
    print(f"wrote {args.out} and {report_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
