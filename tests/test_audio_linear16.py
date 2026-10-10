import numpy as np

from stt_gateway.audio import Linear16Decoder, TARGET_RATE, resample_to_mono_16k


def test_resample_downmixes_and_resamples():
    rate = 48_000
    time = np.arange(rate, dtype=np.float64)
    left = np.sin(2 * np.pi * 440.0 * time / rate).astype(np.float32)
    stereo = np.stack([left, -left], axis=1)  # anti-phase stereo cancels when averaged
    mono = resample_to_mono_16k(stereo, sample_rate=rate, channels=2)
    expected_length = rate * TARGET_RATE // rate
    assert len(mono) == expected_length
    assert float(np.max(np.abs(mono))) < 1e-6


def test_linear16_decoder_yields_16k_mono(sine_seconds, linear16_bytes):
    decoder = Linear16Decoder(sample_rate=48_000, channels=2)
    source = sine_seconds(1.0, 48_000)
    stereo = np.repeat(source.reshape(-1, 1), 2, axis=1)  # interleaved L,R
    assert decoder.drain().size == 0
    decoder.feed(linear16_bytes(stereo.reshape(-1)))
    samples = decoder.drain()
    assert samples.dtype.name == "float32"
    assert abs(len(samples) - TARGET_RATE) < TARGET_RATE * 0.01
    assert float(np.max(np.abs(samples))) > 0.1
    assert decoder.drain().size == 0
    # Partial (non frame-aligned) chunks are buffered until complete.
    decoder.feed(b"\x01\x02")
    assert decoder.drain().size == 0


def test_linear16_decoder_16k_mono_passthrough(sine_seconds, linear16_bytes):
    decoder = Linear16Decoder(sample_rate=16_000, channels=1)
    decoder.feed(linear16_bytes(sine_seconds(0.5)))
    assert decoder.drain().size == 8_000
