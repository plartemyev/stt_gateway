# stt_gateway

Deepgram-compatible streaming speech-to-text WebSocket gateway backed by
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) on CUDA. Drop-in
replacement for `wss://api.deepgram.com/v1/listen` for streaming clients.

> **License warning — read before deploying commercially.** The code is
> GPL-2.0-only, but the default **Armenian** engine, SeamlessM4T v2, uses
> **CC-BY-NC (non-commercial) weights** with additional Meta restrictions.
> No weights are distributed with this project (they are downloaded at
> runtime from Hugging Face), but a commercial deployment that leaves the
> default routing in place would be *using* non-commercial weights — reroute
> `hy` via `STT_ROUTING_CONFIG` if that does not fit your terms. See
> [License & model weights](#license--model-weights).

## Why

A translation pipeline sends microphone/meeting audio to Deepgram's
streaming endpoint. This service mimics exactly that contract locally:

- Same WS path and query parameters: `/v1/listen?encoding=ogg-opus|linear16&sample_rate=48000&channels=2&punctuate=true&model=nova-2&language=en-US|ru|hy-AM|multi`
- Same auth header: `Authorization: Token <any string, max 256 chars>` — any
  value is accepted (length-capped); missing/empty tokens are rejected.
- Same response messages: Deepgram streaming `Results` JSON with
  `channel.alternatives[0].transcript` (the only field the reference
  consumer reads, pinned in `tests/deepgram_consumer_snapshot.py`),
  plus `words`, `confidence`, `metadata` for fidelity with other clients.
- Audio: `ogg-opus` (OGG/WebM Opus container, e.g. Chrome MediaRecorder) and
  `linear16` (headerless s16le PCM at the requested `sample_rate`/`channels`).

Everything is transcribed by warm engines kept loaded for the process
lifetime (see *Model routing* below): a whisper.cpp `whisper-server`
subprocess per configured whisper model (default `ggml-large-v3`, Silero VAD
enabled) and SeamlessM4T v2 large in-process (transformers, fp16 on CUDA),
all with CUDA support.

## Model routing

The API's `model` and `language` query parameters select a warm engine
(configurable via `STT_ROUTING_CONFIG`, a JSON file that replaces the
built-in table — see `stt_gateway/routing.py` for the schema):

| request | engine |
|---|---|
| `model=whisper-large-v3` (or `whisper`, `large-v3`) | whisper large-v3 |
| `model=seamless-m4t-v2-large` (or `seamless`) | SeamlessM4T v2 |
| `model=nova-2` / unknown + `language=hy-AM` | SeamlessM4T v2 (Whisper fails Armenian: ~59% vs 14.4% WER, see `benchmarks/report.md`) |
| `model=nova-2` / unknown + `language=en-US` or `ru` | whisper large-v3 |
| `language=multi` / unspecified | whisper large-v3 (auto-detect) |

Every engine in the table is kept warm; `punctuate=true` (default) guarantees
each utterance-end result closes with sentence-ending punctuation regardless
of engine quirks.

## Known divergences from the real Deepgram

| Aspect | Deepgram | stt_gateway |
|---|---|---|
| Invalid auth status | HTTP 401 on upgrade | HTTP 403 (starlette pre-accept close) |
| `speech_final` cadence | VAD endpointing | emitted with every window; windows are ~`STT_WINDOW_SECONDS` of audio, flushed early on ≥1.2 s trailing silence |
| Interim results | optional `interim_results` | never sent; every message is final |
| `punctuate=false` | disables punctuation | ignored — Whisper output is punctuated natively |
| Unknown `model` values | 400 | accepted and mapped to the configured Whisper model |
| Audio rejected mid-stream | keeps streaming | session closed with WS 1011 |

## Configuration (env, prefix `STT_`)

| Variable | Default | Meaning |
|---|---|---|
| `STT_PORT` | `8030` | gateway listen port |
| `STT_TOKEN_MAX_LEN` | `256` | token length cap |
| `STT_ROUTING_CONFIG` | *(built-in)* | JSON file replacing the engine routing table |
| `STT_MODELS_DIR` | `/models` | where whisper ggml files live (auto-downloaded) |
| `STT_VAD` / `STT_VAD_MODEL` | `1` / `ggml-silero-v6.2.0.bin` | Silero VAD for whisper-server |
| `HF_TOKEN` | *(unset)* | HuggingFace token for model downloads (rate limits) |
| `STT_WINDOW_SECONDS` | `4.0` | transcription window length |
| `STT_FLUSH_SILENCE_SECONDS` | `1.2` | trailing silence that flushes a short window |
| `STT_SILENCE_RMS` | `0.005` | silence energy threshold |
| `STT_MAX_BUFFER_SECONDS` | `300` | buffer cap; beyond it, oldest audio is dropped |
| `STT_ENGINE_*` | see `stt_gateway/config.py` | whisper-server binary/host/port/threads/timeouts |
| `STT_ENGINE_EXTERNAL` | `0` | connect to an already-running whisper-server |
| `STT_DOWNLOAD_MODEL` | `1` | entrypoint downloads missing models from Hugging Face |

## Run

Local (needs a whisper-server at `STT_ENGINE_HOST:STT_ENGINE_PORT`, or set
`STT_ENGINE_EXTERNAL=1` in tests):

```bash
poetry install --with dev
poetry run pytest
poetry run python -m stt_gateway.server
```

Docker (Arch Linux + CUDA 13, NVIDIA driver >= 580 required, GPU pinning
optional):

```bash
docker build -t stt_gateway stt_gateway/
# all GPUs:
docker run --rm --gpus all -p 8030:8030 -v stt_models:/models stt_gateway
# pin one GPU (e.g. the second of two RTX 3090s):
docker run --rm --gpus '"device=1"' -p 8030:8030 -v stt_models:/models stt_gateway
```

docker-compose equivalent:

```yaml
services:
  stt_gateway:
    image: stt_gateway:latest
    restart: unless-stopped
    ports: ["8030:8030"]
    volumes: ["stt_models:/models"]
    environment:
      HF_TOKEN: ${HF_TOKEN}  # optional, for model downloads
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
volumes:
  stt_models:
```

Send a file for debugging (16 kHz mono s16 WAV):

```bash
poetry run python scripts/send_wav.py \
  "ws://127.0.0.1:8030/v1/listen?encoding=linear16&sample_rate=16000&channels=1&language=hy-AM" speech.wav
```

## VRAM usage & performance forecast

Measured on a single RTX 3090 (24 GiB) with both default engines warm
(`nvidia-smi --query-compute-apps`, gateway idle):

| process | VRAM |
|---|---|
| `whisper-server` — ggml-large-v3 + Silero VAD | ~3.8 GiB |
| python — SeamlessM4T v2 fp16 + torch CUDA context | ~3.2 GiB |
| **stt_gateway dual-engine total** | **~7.1 GiB** |

VRAM is per-engine, not per-session: all concurrent sessions share the same
two warm models. Per-session cost is a WS connection, one decoder and a
windower (tens of MiB of host RAM). On a 24 GiB card that leaves ~17 GiB
headroom; the card can even be shared with other GPU services (the reference
deployment colocates stt_gateway with two ~2.1 GiB rendering processes).

### Latency model

A result is emitted when a window closes: either every
`STT_WINDOW_SECONDS` (4 s) of continuous speech, or `flush_silence_seconds`
(1.2 s) after the speaker stops. End-to-end latency for an utterance:

```
latency ≈ utterance_seconds + 1.2 (silence flush) + utterance_seconds × RTF
```

with measured RTFs (FLEURS, RTX 3090 — see `benchmarks/report.md`):

| engine | en | ru | hy |
|---|---|---|---|
| whisper large-v3 (RTF mean) | 0.039 | 0.042 | 0.191 |
| seamless-m4t-v2-large fp16 | ~0.04 | ~0.04 | ~0.04 |

Typical utterance-final latency:

| utterance | en / ru (whisper) | hy (seamless) |
|---|---|---|
| 2 s | ~3.3 s | ~3.3 s |
| 4 s | ~5.4 s | ~5.4 s |

Long continuous speech streams partial results every ~4.2 s (en/ru) with the
sentence completed ~1.4 s after the speaker pauses. `STT_WINDOW_SECONDS` /
`STT_FLUSH_SILENCE_SECONDS` trade latency against whisper accuracy on short
fragments.

### Concurrency forecast

Inference serializes inside each engine, so one GPU sustains roughly
`1 / RTF` concurrent realtime streams per engine:

| concurrent sessions | verdict (single 3090) |
|---|---|
| 1–4 | comfortable, p95 latency ≈ single-session numbers |
| 8 | fine for en/ru (25× realtime headroom); hy sessions queue mildly |
| 16 | en/ru still ~2× headroom; hy/seamless latency grows to ~2× |
| >16–25 | saturated; add a second gateway pinned to the other GPU |

Multi-GPU hosts: run a second `stt_gateway` with
`device_ids: ['1']` (or `--gpus '"device=1"'`) to double capacity; engines
warm in ~2–3 minutes per instance. CPU side (PyAV decode, numpy) stays
negligible up to these numbers.

## Pointing existing Deepgram clients at it

Swap the host in the URL your client builds (e.g. a `DEEPGRAM_WS_BASE`-style
setting or the hardcoded Deepgram URL):

```
wss://api.deepgram.com/v1/listen?...  ->  ws://stt_gateway:8030/v1/listen?...
```

No other changes are required: the `DEEPGRAM_API_KEY` value becomes a dummy
token (any string up to 256 chars).

## Benchmarks

Model quality/latency is verified against the FLEURS test split (en_us,
ru_ru, hy_am — the three pipeline languages) through the production
transcription path. Results: `benchmarks/report.md`.

**Headline finding:** Whisper (turbo and large-v3) handles en/ru well
(6.8-12.3% WER) but fails Armenian (~59% WER). SeamlessM4T v2 large wins all
three languages (en 6.8 / ru 9.9 / hy 14.4% WER) and is ~3x faster — see
`benchmarks/README.md` for the harness and how to reproduce.

```bash
# inside the image (GPU required):
python benchmarks/prepare_data.py --out benchmarks/data --samples 100
python benchmarks/run_bench.py --data benchmarks/data \
    --models ggml-large-v3-turbo ggml-large-v3 seamless-m4t-v2-large \
    --models-dir /models
```

## License & model weights

The code is licensed **GPL-2.0-only** (see `LICENSE`).

**This repository distributes no model weights.** Every engine downloads its
weights at runtime, directly from Hugging Face, into the deploying
instance's own storage — the same pattern as self-hosted object storage:
nothing weight-related passes through this project's releases or image.

| engine | weights | license |
|---|---|---|
| whisper large-v3 (ggml) | `ggerganov/whisper.cpp` on HF | MIT (like Whisper upstream) — commercially usable |
| Silero VAD | `ggml-org/whisper-vad` on HF | MIT — commercially usable |
| SeamlessM4T v2 large | `facebook/seamless-m4t-v2-large` on HF | **CC-BY-NC 4.0 — non-commercial**, plus additional Meta restrictions (e.g. outputs must not be used to train competing models) |

The default routing sends Armenian (`hy`) to SeamlessM4T because it is the
only tested open-weights model that transcribes Armenian reliably (14.4% vs
~59% WER for Whisper — see `benchmarks/report.md`). That default is a
**configuration choice, not a distribution**: a commercial deployment can
reroute `hy` to any commercially usable model by overriding
`STT_ROUTING_CONFIG` and stays non-commercial-clean by doing so. **Whether
the NC terms fit your use of the Seamless weights — including Meta's extra
restrictions on derived models — is the deployer's compliance decision.**

Finding a commercially licensed (ideally CC-BY) Armenian ASR model to make
the default NC-free end to end is the desired future work.
