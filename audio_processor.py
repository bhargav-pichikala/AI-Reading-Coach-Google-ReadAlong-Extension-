"""
audio_processor.py
──────────────────
WebRTC AudioProcessorBase subclass.

Design
------
* recv_queued() accumulates raw PCM frames into a ring-buffer.
* Every time the buffer reaches CHUNK_SECONDS of audio it:
    1. Encodes the PCM to WAV in-memory (no disk I/O).
    2. Submits a transcription job to Groq Whisper via a background
       ThreadPoolExecutor (non-blocking for the WebRTC callback thread).
    3. Appends the returned text to a thread-safe deque that app.py polls.
* Returns the frames unchanged (pass-through) so the browser hears its own mic.
"""

from __future__ import annotations

import io
import wave
import time
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List

import av
import numpy as np
from groq import Groq

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
CHUNK_SECONDS   = 2.0      # seconds of audio per Whisper call
SAMPLE_RATE     = 16_000   # resample target (Whisper optimal)
CHANNELS        = 1
MAX_WORKERS     = 3        # concurrent Whisper requests
MIN_RMS         = 150      # skip silent chunks below this RMS threshold
# ─────────────────────────────────────────────────────────────────────────────


def _frames_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Encode a (N,) int16 numpy array as WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Naive linear resample (fast, no scipy dependency)."""
    if src_rate == dst_rate:
        return pcm
    ratio     = dst_rate / src_rate
    new_len   = int(len(pcm) * ratio)
    indices   = np.linspace(0, len(pcm) - 1, new_len)
    idx_floor = indices.astype(np.int64)
    idx_ceil  = np.clip(idx_floor + 1, 0, len(pcm) - 1)
    frac      = (indices - idx_floor).astype(np.float32)
    resampled = (pcm[idx_floor].astype(np.float32) * (1 - frac) +
                 pcm[idx_ceil ].astype(np.float32) *      frac ).astype(np.int16)
    return resampled


class AudioTranscriptionProcessor:
    """
    Not a subclass of AudioProcessorBase — we use the audio_frame_callback
    pattern instead (simpler, avoids the factory/instance lifecycle issues
    with shared state in Streamlit).

    Usage
    -----
    processor = AudioTranscriptionProcessor(groq_api_key)
    # pass processor.recv_frame as audio_frame_callback to webrtc_streamer
    # poll processor.get_new_text() in the Streamlit loop
    """

    def __init__(self, groq_api_key: str) -> None:
        self._client       = Groq(api_key=groq_api_key)
        self._pcm_buffer   : list[np.ndarray] = []
        self._buffer_len   = 0          # samples accumulated
        self._chunk_samples= int(CHUNK_SECONDS * SAMPLE_RATE)
        self._executor     = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._text_queue   : deque[str] = deque()
        self._lock         = threading.Lock()
        self._src_rate     : int | None = None   # detected on first frame
        self._in_flight    = 0          # Whisper calls in progress
        self._total_chunks = 0

    # ── frame callback (called from WebRTC thread) ────────────────────────
    def recv_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        """
        Receives every audio frame from WebRTC, buffers PCM, fires Whisper
        calls when buffer is full.  Returns frame unchanged (pass-through).
        """
        try:
            # Detect sample rate once
            if self._src_rate is None:
                self._src_rate = frame.sample_rate

            # Convert frame → int16 mono numpy
            audio = frame.to_ndarray()          # shape: (channels, samples) or (samples,)
            if audio.ndim > 1:
                audio = audio.mean(axis=0).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

            # Resample to 16 kHz if needed
            if self._src_rate != SAMPLE_RATE:
                audio = _resample(audio, self._src_rate, SAMPLE_RATE)

            with self._lock:
                self._pcm_buffer.append(audio)
                self._buffer_len += len(audio)

                if self._buffer_len >= self._chunk_samples:
                    chunk = np.concatenate(self._pcm_buffer)
                    self._pcm_buffer  = []
                    self._buffer_len  = 0
                    self._total_chunks += 1
                    self._executor.submit(self._transcribe_chunk, chunk)

        except Exception as exc:
            logger.warning("AudioProcessor recv error: %s", exc)

        return frame

    # ── background transcription ──────────────────────────────────────────
    def _transcribe_chunk(self, pcm: np.ndarray) -> None:
        """Runs in ThreadPoolExecutor — calls Groq Whisper, pushes result."""
        try:
            # Skip silence
            rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
            if rms < MIN_RMS:
                return

            wav_bytes = _frames_to_wav_bytes(pcm, SAMPLE_RATE)
            buf       = io.BytesIO(wav_bytes)
            buf.name  = "audio.wav"

            with self._lock:
                self._in_flight += 1
            try:
                resp = self._client.audio.transcriptions.create(
                    model          = "whisper-large-v3-turbo",
                    file           = buf,
                    response_format= "text",
                    language       = "en",
                )
                text = str(resp).strip()
                if text:
                    self._text_queue.append(text)
            finally:
                with self._lock:
                    self._in_flight -= 1

        except Exception as exc:
            logger.warning("Whisper transcription error: %s", exc)

    # ── public API (called from Streamlit main thread) ────────────────────
    def get_new_text(self) -> list[str]:
        """Drain and return all newly transcribed chunks."""
        results = []
        while self._text_queue:
            results.append(self._text_queue.popleft())
        return results

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    def flush(self) -> None:
        """Force-process any remaining buffered audio (call on stop)."""
        with self._lock:
            if self._pcm_buffer and self._buffer_len > SAMPLE_RATE // 2:
                chunk = np.concatenate(self._pcm_buffer)
                self._pcm_buffer  = []
                self._buffer_len  = 0
                self._executor.submit(self._transcribe_chunk, chunk)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
