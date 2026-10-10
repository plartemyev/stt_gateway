"""Silence-aligned windowing of the decoded audio stream.

The gateway has no interim results to align to, so windows are cut where the
audio is quietest near a target boundary: words survive window edges far
better than at a fixed hard cut. Short buffers flush early once trailing
silence exceeds `flush_silence_seconds` (the speaker stopped talking), which
bounds the added latency after each utterance.
"""

import dataclasses

import numpy as np

from stt_gateway.audio import TARGET_DTYPE, TARGET_RATE


@dataclasses.dataclass(frozen=True)
class Window:
    samples: np.ndarray
    start_seconds: float
    end_seconds: float
    # True when the window was flushed because the speaker stopped talking
    # (trailing silence) rather than because the buffer reached its target
    # size; drives the punctuate=true terminal-punctuation rule.
    utterance_end: bool = False


class Windower:
    """Accumulates 16 kHz mono samples and slices out transcription windows."""

    FRAME_SAMPLES = 320  # 20 ms at 16 kHz

    def __init__(
            self,
            *,
            window_seconds: float,
            cut_lookback_seconds: float,
            flush_silence_seconds: float,
            min_utterance_seconds: float,
            max_buffer_seconds: float,
            silence_rms: float,
    ):
        self.window_seconds = window_seconds
        self.cut_lookback_seconds = cut_lookback_seconds
        self.flush_silence_seconds = flush_silence_seconds
        self.min_utterance_seconds = min_utterance_seconds
        self.max_buffer_seconds = max_buffer_seconds
        self.silence_rms = silence_rms
        self._buffer = np.empty(0, dtype=TARGET_DTYPE)
        self._consumed_seconds = 0.0
        self.dropped_seconds = 0.0

    @property
    def buffered_seconds(self) -> float:
        """Seconds of audio currently held back from transcription."""
        return len(self._buffer) / TARGET_RATE

    @property
    def consumed_seconds(self) -> float:
        """Stream time already emitted or dropped; the start of the next window."""
        return self._consumed_seconds

    def push(self, samples: np.ndarray) -> None:
        """Append decoded 16 kHz mono samples, trimming the oldest on overflow."""
        if len(samples) == 0:
            return
        self._buffer = np.concatenate((self._buffer, samples.astype(TARGET_DTYPE, copy=False)))
        limit = int(self.max_buffer_seconds * TARGET_RATE)
        overflow = len(self._buffer) - limit
        if overflow > 0:
            self._buffer = self._buffer[overflow:]
            self._consumed_seconds += overflow / TARGET_RATE
            self.dropped_seconds += overflow / TARGET_RATE

    def pop_window(self) -> Window | None:
        """Return the next window to transcribe, or None.

        When the buffer holds a full window the cut lands on the quietest
        20 ms frame near the target boundary. Otherwise the whole buffer is
        flushed early once it ends in enough silence.
        """
        window_samples = int(self.window_seconds * TARGET_RATE)
        if len(self._buffer) >= window_samples:
            return self._take(self._silence_aligned_cut(window_samples))
        min_utterance_samples = int(self.min_utterance_seconds * TARGET_RATE)
        if len(self._buffer) >= min_utterance_samples:
            if self._trailing_silence_seconds() >= self.flush_silence_seconds:
                return self._take(len(self._buffer), utterance_end=True)
        return None

    def _take(self, cut_samples: int, utterance_end: bool = False) -> Window:
        start = self._consumed_seconds
        end = start + cut_samples / TARGET_RATE
        samples = self._buffer[:cut_samples].copy()
        self._buffer = self._buffer[cut_samples:]
        self._consumed_seconds = end
        return Window(samples, start, end, utterance_end)

    def _silence_aligned_cut(self, target_samples: int) -> int:
        lookback = int(self.cut_lookback_seconds * TARGET_RATE)
        low = max(target_samples - lookback, self.FRAME_SAMPLES)
        high = min(target_samples + lookback, len(self._buffer))
        frames = self._frame_rms()
        first_frame = low // self.FRAME_SAMPLES
        last_frame = max(high // self.FRAME_SAMPLES, first_frame + 1)
        quietest = first_frame + int(np.argmin(frames[first_frame:last_frame]))
        return min((quietest + 1) * self.FRAME_SAMPLES, len(self._buffer))

    def _frame_rms(self) -> np.ndarray:
        count = len(self._buffer) // self.FRAME_SAMPLES
        frames = self._buffer[: count * self.FRAME_SAMPLES].reshape(count, self.FRAME_SAMPLES)
        return np.sqrt(np.mean(np.square(frames), axis=1))

    def _trailing_silence_seconds(self) -> float:
        frames = self._frame_rms()
        quiet = 0
        for index in range(len(frames) - 1, -1, -1):
            if frames[index] >= self.silence_rms:
                break
            quiet += 1
        return quiet * self.FRAME_SAMPLES / TARGET_RATE
