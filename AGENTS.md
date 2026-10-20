# Agent guidelines: stt_gateway

Deepgram-compatible streaming speech-to-text WebSocket gateway backed by
whisper.cpp (CUDA) and SeamlessM4T v2. Published as GPL-2.0-only.

## Tooling

- Package management is **Poetry**: `poetry lock` / `poetry install --with dev`
  (in-project `.venv`). The Docker image installs poetry via pip
  (`PIP_BREAK_SYSTEM_PACKAGES=1`; the Arch base is PEP-668 externally
  managed) and runs `poetry install --only main`.
- Lint with flake8 (config in `.flake8`,
  `application_import_names = stt_gateway`; per-file-ignores cover
  `tests/*`, `benchmarks/*`, `scripts/*` — operator tooling uses
  prints-as-logging and pinned-binary subprocess calls).
- Docker build uses the current Arch repo `cuda` (13.x) — no archive pins.
  Both whisper.cpp (cu13 runtime) and torch cu13 wheels require
  **NVIDIA driver >= 580** on the GPU host; do not reintroduce CUDA 12
  pins for older drivers.

## Engine contract (whisper.cpp `whisper-server`, verified against sources)

- Readiness probe: any HTTP answer on `GET /`.
- Inference: `POST /inference` multipart with per-request override fields
  `language` and `response_format=verbose_json`.
- `verbose_json` exposes `segments[].{text,start,end,avg_logprob}`; word
  entries are sub-word token fragments — the gateway rebuilds words by
  splitting segment text.

## Dual-engine routing

- Default table lives in `stt_gateway/routing.py`; a JSON file
  (`STT_ROUTING_CONFIG`) replaces it wholesale.
- Warm engines: whisper large-v3 (whisper-server subprocess, ports from
  `STT_ENGINE_PORT` up) + SeamlessM4T v2 fp16 in-process. All engines named
  in the table stay loaded; sessions route per request.
- Seamless specifics (transformers 5.x): no `task=` kwarg on the
  `ForSpeechToText` class, `sentencepiece` required for its tokenizer,
  `revision` pinned in the routing table, snapshot download scoped with
  `allow_patterns` (legacy `.pt` checkpoints + vocoder are never read).

## Gotchas

- PyAV (`av.open` on a pipe) blocks forever waiting for probe bytes on
  unknown input — never feed it unvalidated data;
  `stt_gateway.audio.ContainerDecoder` checks the OggS/EBML magic before the
  first write to the pipe.
- Silero VAD ggml model for whisper.cpp lives in the HF repo
  `ggml-org/whisper-vad` (NOT `ggml-org/whisper.cpp`):
  `https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin`.
- Deepgram streaming WS messages carry the transcript at top-level
  `channel.alternatives[0].transcript` (unlike the prerecorded HTTP API,
  which nests under `results.channels`); the pinned consumer snapshot in
  `tests/deepgram_consumer_snapshot.py` reads exactly that path.
- Never commit tokens: pass `HF_TOKEN` and friends via env / `--env-file`.
- Capacity on one RTX 3090: dual warm engines ≈ 7.1 GiB VRAM (whisper
  large-v3 3.8 + Seamless fp16 3.2), ~25× realtime per engine — measured
  numbers and the concurrency/latency forecast live in `README.md`
  ("VRAM usage & performance forecast").

## Licensing landmine: Seamless weights are CC-BY-NC

The code is GPL-2.0-only, but the **SeamlessM4T v2 weights are CC-BY-NC
4.0** — non-commercial, with additional Meta restrictions (e.g. outputs must
not be used to train competing models). NC terms also conflict with GPL's
no-further-restrictions distribution rule, so:

- **Never ship the weights** — not in git, not baked into the Docker image,
  not mirrored in releases. They are downloaded at runtime by the deploying
  instance straight from Hugging Face; nothing NC is distributed with this
  code.
- The default Armenian (`hy`) routing target is Seamless. Commercial
  deployers must reroute `hy` via `STT_ROUTING_CONFIG` to a commercially
  usable model — that compliance decision is theirs, and the README says so
  loudly.
- Keep this pattern intact in any refactor: runtime download + config-only
  routing. If a CC-BY (commercially usable) Armenian ASR model surfaces,
  switching the default `hy` target to it removes the issue entirely — that
  is the desired end state.
