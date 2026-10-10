#!/usr/bin/env python
r"""Debug client: stream a WAV file into the gateway the way a Deepgram client does.

The WAV must already be 16 kHz mono s16 (convert with ffmpeg if needed):
    ffmpeg -i input.any -ac 1 -ar 16000 -c:a pcm_s16le output.wav

Usage:
    python scripts/send_wav.py ws://127.0.0.1:8030/v1/listen?encoding=linear16\\
&sample_rate=16000&channels=1&language=en-US audio.wav [chunk_seconds]
"""

import asyncio
import contextlib
import sys
import wave

import websockets


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    url, wav_path = sys.argv[1], sys.argv[2]
    chunk_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    with wave.open(wav_path, "rb") as wav_file:
        if wav_file.getframerate() != 16000 or wav_file.getnchannels() != 1:
            raise SystemExit("WAV must be 16 kHz mono; see the ffmpeg hint in --help")
        chunk_samples = int(16000 * chunk_seconds)
        frame_bytes = 2 * chunk_samples
        async with websockets.connect(url, additional_headers={"Authorization": "Token local-dev"}) as ws:
            hello = await ws.recv()
            print(f"hello: {hello}")
            while True:
                frames = wav_file.readframes(chunk_samples)
                if not frames:
                    break
                await ws.send(frames)
                if len(frames) < frame_bytes:
                    break
                with contextlib.suppress(asyncio.TimeoutError):
                    result = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    print(f"result: {result}")
            while True:
                try:
                    print(f"result: {await asyncio.wait_for(ws.recv(), timeout=15.0)}", flush=True)
                except asyncio.TimeoutError:
                    print("no more results; exiting")
                    return
                except websockets.ConnectionClosed as e:
                    print(f"closed: code={e.rcvd.code if e.rcvd else None}")
                    return


if __name__ == "__main__":
    asyncio.run(main())
