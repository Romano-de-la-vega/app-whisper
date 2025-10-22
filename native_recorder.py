"""Utilities to capture system audio from a native Python process.

The goal of this module is to expose a small helper that records the main
system output (loopback) without going through browser APIs.  The
implementation currently targets Windows with WASAPI loopback support via the
``soundcard`` package.  Other platforms raise ``NativeRecorderUnsupported``.

The recorder is intentionally simple: only a single recording session can be
active at a time.  When a session is started we spawn a background reader that
pulls audio blocks from the loopback device into a queue.  A writer thread
consumes the queue and stores the samples in a ``.wav`` file using
``soundfile``.  When the user stops the recording the worker threads are
stopped gracefully and the resulting file path + duration are returned.

This module is designed so it can be safely imported even if the optional
dependencies are missing – ``NativeRecorder`` will simply report that native
recording is unavailable.  FastAPI routes can use this information to decide
whether to expose UI elements for the native recorder.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


try:  # Optional dependencies used only when native recording is available.
    import numpy as np
except Exception:  # pragma: no cover - handled gracefully at runtime
    np = None  # type: ignore
else:
    if not getattr(np, "_soundcard_fromstring_patch", False):
        _np_fromstring_original = np.fromstring  # type: ignore[attr-defined]

        def _soundcard_fromstring_compat(
            data, dtype=float, count=-1, sep=""
        ):  # type: ignore[override]
            if sep == "":
                try:
                    return np.frombuffer(data, dtype=dtype, count=count)
                except TypeError:
                    pass
            return _np_fromstring_original(data, dtype=dtype, count=count, sep=sep)

        np.fromstring = _soundcard_fromstring_compat  # type: ignore[attr-defined]
        np._soundcard_fromstring_patch = True  # type: ignore[attr-defined]

try:
    import soundcard as sc
except Exception:  # pragma: no cover - handled gracefully at runtime
    sc = None  # type: ignore

try:
    import soundfile as sf
except Exception:  # pragma: no cover - handled gracefully at runtime
    sf = None  # type: ignore


class NativeRecorderError(RuntimeError):
    """Base class for native recorder exceptions."""


class NativeRecorderUnsupported(NativeRecorderError):
    """Raised when the current platform/backend cannot capture system audio."""


class NativeRecorderBusy(NativeRecorderError):
    """Raised when attempting to start a second recording while one is active."""


class NativeRecorderNotRunning(NativeRecorderError):
    """Raised when requesting to stop a recording but none is running."""


@dataclass
class NativeRecordingResult:
    """Metadata returned once a recording session finishes."""

    recording_id: str
    path: Path
    duration: float


class _RecordingSession:
    """Internal helper streaming audio blocks from the system loopback."""

    def __init__(self, recording_id: str, output_path: Path, device_id: str, samplerate: float, channels: int):
        if sc is None or sf is None or np is None:
            raise NativeRecorderUnsupported("Le module soundcard/soundfile est indisponible.")

        self.recording_id = recording_id
        self.output_path = output_path
        self._device_id = device_id
        self._samplerate = float(samplerate)
        self._channels = int(max(1, channels))
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._recorder_ctx: Optional[object] = None
        self._recorder: Optional[object] = None
        self._channel_indices: list[int] = []
        self._blocksize = 2048
        self._start_ts: Optional[float] = None
        self._duration: float = 0.0

    # ---- public API -----------------------------------------------------
    def start(self) -> None:
        """Start streaming audio data to ``output_path``."""

        if sc is None or sf is None or np is None:
            raise NativeRecorderUnsupported("Le module soundcard/soundfile est indisponible.")

        try:
            microphone = sc.get_microphone(id=self._device_id, include_loopback=True)
        except Exception as exc:  # pragma: no cover - depends on host runtime
            raise NativeRecorderUnsupported("Impossible d'ouvrir l'entrée loopback WASAPI.") from exc

        available_channels = int(getattr(microphone, "channels", 0) or 0)
        if available_channels <= 0:
            raise NativeRecorderUnsupported("Le périphérique loopback ne propose aucun canal audio.")

        self._channel_indices = list(range(min(self._channels, available_channels)))
        if not self._channel_indices:
            raise NativeRecorderUnsupported("Le périphérique loopback ne propose aucun canal audio exploitable.")
        self._channels = len(self._channel_indices)

        self._recorder_ctx = microphone.recorder(
            samplerate=int(self._samplerate),
            channels=self._channel_indices,
            blocksize=self._blocksize,
            exclusive_mode=False,
        )
        self._recorder = self._recorder_ctx.__enter__()
        self._stop_event.clear()
        self._queue = queue.Queue()
        self._start_ts = time.monotonic()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def stop(self) -> NativeRecordingResult:
        """Stop the session and return recording metadata."""

        self._stop_event.set()

        if np is not None:
            try:
                self._queue.put_nowait(np.empty((0, self._channels), dtype="float32"))
            except Exception:  # pragma: no cover - best effort
                pass

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)
            self._reader_thread = None

        if self._recorder_ctx is not None:
            try:
                self._recorder_ctx.__exit__(None, None, None)
            finally:
                self._recorder_ctx = None
                self._recorder = None

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5)
            self._writer_thread = None

        if self._start_ts is not None:
            self._duration = max(0.0, time.monotonic() - self._start_ts)
            self._start_ts = None

        return NativeRecordingResult(
            recording_id=self.recording_id,
            path=self.output_path,
            duration=self._duration,
        )

    # ---- internals ------------------------------------------------------
    def _reader_loop(self) -> None:
        if self._recorder is None or np is None:  # pragma: no cover - guarded in start
            return

        try:
            while not self._stop_event.is_set():
                chunk = self._recorder.record(self._blocksize)
                if chunk is None or getattr(chunk, "size", 0) == 0:
                    continue
                self._queue.put(np.asarray(chunk, dtype="float32"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[NativeRecorder] reader error: {exc}")
        finally:
            self._stop_event.set()

    def _writer_loop(self) -> None:
        if sf is None or np is None:  # pragma: no cover - guarded on start
            return

        # ``soundfile`` expects ints for PCM_16, hence the explicit conversion.
        with sf.SoundFile(
            self.output_path,
            mode="w",
            samplerate=int(self._samplerate),
            channels=self._channels,
            subtype="PCM_16",
        ) as file:
            while True:
                try:
                    data = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue

                if getattr(data, "size", 0) == 0:
                    if self._stop_event.is_set():
                        break
                    continue

                file.write((np.clip(data, -1.0, 1.0) * 32767).astype("int16"))

            # Drain any remaining buffers after stop to avoid truncation.
            while not self._queue.empty():
                leftover = self._queue.get()
                if getattr(leftover, "size", 0):
                    file.write((np.clip(leftover, -1.0, 1.0) * 32767).astype("int16"))


class NativeRecorder:
    """Coordinator object managing native recording sessions."""

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active: Optional[_RecordingSession] = None
        self._completed: Dict[str, NativeRecordingResult] = {}

    # ---- public helpers --------------------------------------------------
    def is_available(self) -> bool:
        """Return True if the current environment supports native recording."""

        if sc is None or sf is None or np is None:
            return False
        try:
            self._pick_device()
        except NativeRecorderUnsupported:
            return False
        except Exception:
            return False
        return True

    def start(self) -> NativeRecordingResult:
        """Start a new native recording session."""

        with self._lock:
            if self._active is not None:
                raise NativeRecorderBusy("Un enregistrement est déjà en cours.")

            device_id, samplerate, channels = self._pick_device()
            recording_id = uuid.uuid4().hex
            output_path = self._base_dir / f"native_recording_{recording_id}.wav"
            session = _RecordingSession(recording_id, output_path, device_id, samplerate, channels)
            session.start()
            self._active = session
            self._purge_old_files_locked()
            return NativeRecordingResult(recording_id=recording_id, path=output_path, duration=0.0)

    def stop(self, recording_id: Optional[str] = None) -> NativeRecordingResult:
        """Stop the active recording session."""

        with self._lock:
            if self._active is None:
                raise NativeRecorderNotRunning("Aucun enregistrement natif en cours.")

            session = self._active
            if recording_id and recording_id != session.recording_id:
                raise NativeRecorderNotRunning("Identifiant d'enregistrement invalide.")

            result = session.stop()
            self._active = None
            self._completed[result.recording_id] = result
            return result

    def get_completed(self, recording_id: str) -> Optional[NativeRecordingResult]:
        with self._lock:
            return self._completed.get(recording_id)

    # ---- helpers ---------------------------------------------------------
    def _pick_device(self) -> tuple[str, float, int]:
        if sc is None:
            raise NativeRecorderUnsupported("Le module soundcard n'est pas installé.")

        def _candidate_from_speaker(speaker) -> Optional[tuple[str, int]]:
            try:
                microphone = sc.get_microphone(id=getattr(speaker, "id", None), include_loopback=True)
            except Exception:
                try:
                    microphone = sc.get_microphone(id=getattr(speaker, "name", ""), include_loopback=True)
                except Exception:
                    return None
            channels = int(getattr(microphone, "channels", None) or getattr(speaker, "channels", 0) or 0)
            if channels <= 0:
                return None
            return microphone.id, channels

        # Prefer the default speaker when available.
        try:
            default_speaker = sc.default_speaker()
        except Exception:
            default_speaker = None

        if default_speaker is not None:
            candidate = _candidate_from_speaker(default_speaker)
            if candidate:
                device_id, channels = candidate
                return device_id, 48000.0, max(1, channels)

        try:
            speakers = sc.all_speakers()
        except Exception as exc:  # pragma: no cover - depends on host runtime
            raise NativeRecorderUnsupported("Impossible de lister les périphériques audio.") from exc

        best_candidate: Optional[tuple[str, int]] = None
        for speaker in speakers:
            candidate = _candidate_from_speaker(speaker)
            if not candidate:
                continue
            if best_candidate is None or candidate[1] > best_candidate[1]:
                best_candidate = candidate

        if best_candidate is None:
            raise NativeRecorderUnsupported("Aucun périphérique de sortie compatible loopback n'a été trouvé.")

        device_id, channels = best_candidate
        return device_id, 48000.0, max(1, channels)

    def _purge_old_files_locked(self, max_age_seconds: float = 3600.0) -> None:
        now = time.time()
        to_delete = []
        for rec_id, result in list(self._completed.items()):
            if not result.path.exists():
                to_delete.append(rec_id)
                continue
            if now - result.path.stat().st_mtime > max_age_seconds:
                to_delete.append(rec_id)

        for rec_id in to_delete:
            result = self._completed.pop(rec_id, None)
            if result and result.path.exists():
                try:
                    result.path.unlink()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

