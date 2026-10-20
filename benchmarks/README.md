# stt_gateway benchmarks

Model quality/latency verification for the three pipeline languages
(English, Russian, Armenian) on the FLEURS test split, run on any CUDA GPU
host with Docker and the NVIDIA container runtime (measured on RTX 3090).

## Models

| model | path | notes |
|---|---|---|
| `ggml-large-v3-turbo` | whisper.cpp, CUDA | faster distilled candidate, benched |
| `ggml-large-v3` | whisper.cpp, CUDA (production engine) | accuracy reference |
| `seamless-m4t-v2-large` | HF transformers, in-process | open-weights multilingual reference (en/ru/hy); **CC-BY-NC weights** — benchmark only, never redistribute |
| Google Chirp 2/3 | GCP STT API v2 only | skipped: no public weights, needs GCP credentials |

## Metrics

- **WER %** — jiwer, normalized text (lowercase, punctuation stripped).
- **RTF** (real-time factor) — transcription wall time / audio duration;
  <1.0 means faster than real time. ggml models: warm whisper-server HTTP
  round-trip (production path, Silero VAD on). Seamless: in-process
  `generate()`, greedy decoding, no punctuation model.
- Model load time is reported separately.

## Run

All commands run on the GPU host; the workspace (below `~/stt_gateway_build`)
holds a synced copy of this repository plus `models/`, `data/` and `hf/`
(HuggingFace cache) volumes.

```bash
# 1. build the image (see Dockerfile; ~15 min, CUDA layers cached afterwards)
~/stt_gateway_build/build.sh

# 2. download model weights (idempotent)
~/stt_gateway_build/download_models.sh      # whisper ggml + silero VAD
~/stt_gateway_build/download_seamless.sh    # facebook/seamless-m4t-v2-large -> hf/

# 3. FLEURS data prep: 100 samples per language -> data/*.wav + *.jsonl
~/stt_gateway_build/prepare.sh              # CPU-only; poetry dev deps inside

# 4. WER + RTF for all models on a free GPU
~/stt_gateway_build/bench.sh                # -> data/results.json + data/report.md

# 5. copy the report back into the repository
scp <gpu-host>:~/stt_gateway_build/data/report.md benchmarks/report.md
```

The workspace scripts referenced above are thin `docker run` wrappers;
replicate them from the Run section of the main README if you prefer to
drive Docker directly. Sample sizes are configurable (`--samples N`,
default 100 per language). `prepare.py` pins the FLEURS revision for
reproducibility.
