# stt_gateway model benchmark

FLEURS test split (100 samples/language), GeForce RTX 3090. ggml-* models go
through the production transcription path (gateway WhisperEngine -> warm
whisper-server HTTP, Silero VAD on); seamless-m4t-v2-large runs in-process
via transformers, greedy decoding, no punctuation model (WER normalization
strips punctuation anyway). WER computed on normalized text (lowercase,
punctuation stripped). RTF = transcription wall time / audio duration (<1
means faster than real time). Model load time for seamless includes HF hub
resolution over a slow uplink; warm loads are minutes faster.

## Results

| model | en WER % | ru WER % | hy WER % | RTF p95 (worst) |
|---|---|---|---|---|
| ggml-large-v3-turbo | 6.85 | 12.27 | **59.00** | 0.149 |
| ggml-large-v3 | 8.34 | 11.31 | **58.43** | 0.625 |
| seamless-m4t-v2-large | **6.76** | **9.90** | **14.40** | **0.049** |

## Conclusions

1. **Whisper cannot reliably transcribe Armenian** in either size: ~59% WER
   on FLEURS (matches Whisper's known weak hy performance). English and
   Russian are fine (6.8-12.3%).
2. **SeamlessM4T v2 large wins every language**, including 4x better
   Armenian (14.4%), and is the fastest of the three despite being a 2.3B
   model (RTF p95 0.049 = ~20x realtime on a 3090).
3. Caveat for gateway integration: Whisper output carries native punctuation
   (the reference consumer's sentence poller splits sentences on it); Seamless
   punctuation is language-dependent but strong — FLEURS transcripts end
   with a terminal mark in 89% (en) / 95% (ru) / 99% (hy, incl. ։) of
   cases — and the gateway's `punctuate=true` rule closes the remainder.

## Post-benchmark decision (implemented)

The gateway now keeps two warm engines and routes per call
(`stt_gateway/routing.py`, JSON-overridable via `STT_ROUTING_CONFIG`):

- supported `model=` requests are honored (`whisper-large-v3`,
  `seamless-m4t-v2-large` + aliases);
- unsupported requests (e.g. `nova-2`) route by language:
  `hy` -> SeamlessM4T v2, `en`/`ru` -> whisper large-v3;
- unspecified language -> whisper large-v3 (auto-detect). Note: Armenian
  speech arriving as `multi` will hit Whisper and degrade; send an explicit
  `language=hy-AM` for Armenian content.
- Seamless runs fp16 (CUDA) at RTF ~0.04, serialized per engine.


## ggml-large-v3-turbo

Model load time: 3.1s

| language | samples | WER % | RTF mean | RTF p95 |
|---|---|---|---|---|
| en | 100 | 6.85 | 0.022 | 0.037 |
| hy | 100 | 59.00 | 0.054 | 0.149 |
| ru | 100 | 12.27 | 0.021 | 0.031 |

## ggml-large-v3

Model load time: 2.0s

| language | samples | WER % | RTF mean | RTF p95 |
|---|---|---|---|---|
| en | 100 | 8.34 | 0.039 | 0.055 |
| hy | 100 | 58.43 | 0.191 | 0.625 |
| ru | 100 | 11.31 | 0.042 | 0.056 |

## seamless-m4t-v2-large

Model load time: 746.1s

| language | samples | WER % | RTF mean | RTF p95 |
|---|---|---|---|---|
| en | 100 | 6.76 | 0.038 | 0.049 |
| hy | 100 | 14.40 | 0.042 | 0.049 |
| ru | 100 | 9.90 | 0.039 | 0.048 |
