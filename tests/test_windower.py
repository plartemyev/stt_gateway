import numpy as np

from stt_gateway.windower import Windower


def make_windower(**overrides) -> Windower:
    params = {
        "window_seconds": 4.0,
        "cut_lookback_seconds": 0.75,
        "flush_silence_seconds": 1.2,
        "min_utterance_seconds": 0.2,
        "max_buffer_seconds": 300.0,
        "silence_rms": 0.005,
    }
    params.update(overrides)
    return Windower(**params)


def test_below_min_utterance_never_emits():
    windower = make_windower()
    windower.push(np.zeros(1600, dtype=np.float32))  # 0.1s
    assert windower.pop_window() is None


def test_full_window_is_cut_at_silence_aligned_boundary():
    windower = make_windower()
    windower.push(np.ones(5 * 16_000, dtype=np.float32))  # 5s uniform energy
    window = windower.pop_window()
    assert window is not None
    assert window.start_seconds == 0.0
    assert window.utterance_end is False  # boundary cut, not an utterance end
    # Cut lands within lookback of the 4s target on a 20ms frame boundary.
    assert abs(window.end_seconds - 4.0) <= 0.75
    assert window.end_seconds > 0
    remainder = windower.buffered_seconds
    assert abs(remainder - (5.0 - window.end_seconds)) < 0.01
    assert abs(windower.consumed_seconds - window.end_seconds) < 0.01


def test_trailing_silence_flushes_short_buffer():
    windower = make_windower(flush_silence_seconds=0.2)
    loud = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 200, 8_000)).astype(np.float32)
    windower.push(np.concatenate([loud, np.zeros(4_000, dtype=np.float32)]))
    window = windower.pop_window()
    assert window is not None
    assert abs(window.end_seconds - 0.75) < 0.01
    assert window.utterance_end is True  # speaker stopped talking
    assert windower.buffered_seconds == 0.0


def test_trailing_speech_below_flush_silence_is_kept():
    windower = make_windower(flush_silence_seconds=1.0)
    loud = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 200, 16_000)).astype(np.float32)
    windower.push(loud)  # 1s of speech, no silence
    assert windower.pop_window() is None


def test_consecutive_windows_keep_absolute_timing():
    windower = make_windower()
    windower.push(np.ones(5 * 16_000, dtype=np.float32))
    first = windower.pop_window()
    windower.push(np.ones(5 * 16_000, dtype=np.float32))
    second = windower.pop_window()
    assert second.start_seconds == first.end_seconds


def test_buffer_cap_drops_oldest_and_advances_time():
    windower = make_windower(max_buffer_seconds=1.0)
    windower.push(np.ones(2 * 16_000, dtype=np.float32))
    assert windower.buffered_seconds <= 1.0
    assert windower.consumed_seconds >= 1.0
    assert windower.dropped_seconds >= 1.0
