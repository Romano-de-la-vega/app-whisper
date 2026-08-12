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
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_MAX_CAPTURE_CHANNELS = 2
_QUEUE_MAX_BLOCKS = 64
_QUEUE_PUT_TIMEOUT_SECONDS = 0.1
_THREAD_STOP_TIMEOUT_SECONDS = 5.0
_READER_STOP_GRACE_SECONDS = 0.5
_RECORDING_NAME_RE = re.compile(r"native_recording_([0-9a-f]{32})\.wav\Z")


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
        self._channels = min(_MAX_CAPTURE_CHANNELS, int(max(1, channels)))
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
        self._stop_event = threading.Event()
        self._reader_done_event = threading.Event()
        self._writer_ready_event = threading.Event()
        self._writer_done_event = threading.Event()
        self._recorder_closing_event = threading.Event()
        self._error_lock = threading.Lock()
        self._worker_failure: Optional[tuple[str, BaseException]] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._recorder_ctx: Optional[object] = None
        self._recorder: Optional[object] = None
        self._channel_indices: list[int] = []
        self._blocksize = 2048
        self._start_ts: Optional[float] = None
        self._duration: float = 0.0
        self._frames_written: int = 0

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

        self._stop_event.clear()
        self._reader_done_event.clear()
        self._writer_ready_event.clear()
        self._writer_done_event.clear()
        self._recorder_closing_event.clear()
        with self._error_lock:
            self._worker_failure = None
        self._queue = queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
        self._frames_written = 0
        self._duration = 0.0
        self._start_ts = time.monotonic()

        try:
            recorder_ctx = microphone.recorder(
                samplerate=int(self._samplerate),
                channels=self._channel_indices,
                blocksize=self._blocksize,
                exclusive_mode=False,
            )
            recorder = recorder_ctx.__enter__()
            self._recorder_ctx = recorder_ctx
            self._recorder = recorder

            # Open the output first. This prevents a producer from filling the
            # queue when the destination cannot be created.
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
            if not self._writer_ready_event.wait(timeout=_THREAD_STOP_TIMEOUT_SECONDS):
                raise NativeRecorderError("Le thread d'écriture audio ne s'est pas initialisé à temps.")
            self._raise_worker_failure()

            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
        except Exception:
            self._rollback_failed_start()
            raise

    def stop(self) -> NativeRecordingResult:
        """Stop the session and return recording metadata."""

        self._stop_event.set()
        deadline = time.monotonic() + _THREAD_STOP_TIMEOUT_SECONDS
        close_error: Optional[BaseException] = None

        reader = self._reader_thread
        if reader is None or not reader.is_alive():
            self._reader_done_event.set()
        else:
            reader.join(timeout=min(_READER_STOP_GRACE_SECONDS, self._remaining(deadline)))

        # A device failure can leave record() blocked. Closing the WASAPI
        # context after a short grace period is the only reliable unblock.
        try:
            self._close_recorder_context()
        except Exception as exc:  # keep joining workers before surfacing it
            close_error = exc

        if reader is not None and reader.is_alive():
            reader.join(timeout=self._remaining(deadline))

        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            writer.join(timeout=self._remaining(deadline))

        stuck_workers = []
        if reader is not None and reader.is_alive():
            stuck_workers.append("capture")
        else:
            self._reader_thread = None
        if writer is not None and writer.is_alive():
            stuck_workers.append("écriture")
        else:
            self._writer_thread = None

        self._start_ts = None

        if stuck_workers:
            names = " et ".join(stuck_workers)
            raise NativeRecorderError(f"Arrêt incomplet : le thread de {names} est toujours actif.")
        if close_error is not None:
            raise NativeRecorderError("Impossible de fermer proprement le périphérique audio.") from close_error

        self._raise_worker_failure()
        frames_on_disk = self._validate_output_file()
        self._duration = frames_on_disk / max(self._samplerate, 1.0)

        return NativeRecordingResult(
            recording_id=self.recording_id,
            path=self.output_path,
            duration=self._duration,
        )

    # ---- internals ------------------------------------------------------
    def _reader_loop(self) -> None:
        try:
            if self._recorder is None or np is None:  # pragma: no cover - guarded in start
                raise NativeRecorderError("Le périphérique de capture n'est pas initialisé.")

            while not self._stop_event.is_set():
                chunk = self._recorder.record(self._blocksize)
                if chunk is None or getattr(chunk, "size", 0) == 0:
                    continue
                data = np.asarray(chunk, dtype="float32")

                # Do not discard a block already returned by record(), even if
                # stop() was requested meanwhile. Backpressure remains bounded
                # and the writer is the only condition that can make us abort.
                while True:
                    if self._writer_done_event.is_set():
                        break
                    try:
                        self._queue.put(data, timeout=_QUEUE_PUT_TIMEOUT_SECONDS)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            # __exit__ may intentionally interrupt a blocked record() during
            # stop(); that exception is not a capture failure.
            if not self._recorder_closing_event.is_set():
                self._record_worker_failure("capture", exc)
        finally:
            self._reader_done_event.set()
            self._stop_event.set()

    def _writer_loop(self) -> None:
        try:
            if sf is None or np is None:  # pragma: no cover - guarded on start
                raise NativeRecorderUnsupported("Le module soundfile/numpy est indisponible.")

            # ``soundfile`` expects ints for PCM_16, hence the explicit conversion.
            with sf.SoundFile(
                self.output_path,
                mode="w",
                samplerate=int(self._samplerate),
                channels=self._channels,
                subtype="PCM_16",
            ) as file:
                self._writer_ready_event.set()
                while True:
                    try:
                        data = self._queue.get(timeout=0.2)
                    except queue.Empty:
                        if self._reader_done_event.is_set():
                            break
                        continue

                    if getattr(data, "size", 0) == 0:
                        continue

                    array = np.asarray(data)
                    if array.ndim == 1:
                        if self._channels != 1:
                            raise RuntimeError("Bloc mono reçu pour une capture multicanal.")
                        frame_count = int(array.shape[0])
                    elif array.ndim == 2:
                        if int(array.shape[1]) != self._channels:
                            raise RuntimeError(
                                f"Nombre de canaux inattendu ({array.shape[1]} au lieu de {self._channels})."
                            )
                        frame_count = int(array.shape[0])
                    else:
                        raise RuntimeError("Format de bloc audio inattendu.")

                    if frame_count <= 0:
                        continue
                    encoded = (np.clip(array, -1.0, 1.0) * 32767).astype("int16")
                    file.write(encoded)
                    self._frames_written += frame_count
        except Exception as exc:
            self._record_worker_failure("écriture", exc)
        finally:
            # start() waits on ready even when opening the file failed.
            self._writer_ready_event.set()
            self._writer_done_event.set()
            self._stop_event.set()

    def _record_worker_failure(self, role: str, exc: BaseException) -> None:
        with self._error_lock:
            if self._worker_failure is None:
                self._worker_failure = (role, exc)

    def _raise_worker_failure(self) -> None:
        with self._error_lock:
            failure = self._worker_failure
        if failure is None:
            return
        role, exc = failure
        raise NativeRecorderError(f"Erreur du thread de {role} audio : {exc}") from exc

    def _close_recorder_context(self) -> None:
        context = self._recorder_ctx
        if context is None:
            return
        self._recorder_closing_event.set()
        try:
            context.__exit__(None, None, None)
        finally:
            self._recorder_ctx = None
            self._recorder = None

    def _rollback_failed_start(self) -> None:
        self._stop_event.set()
        if self._reader_thread is None or not self._reader_thread.is_alive():
            self._reader_done_event.set()
        try:
            self._close_recorder_context()
        except Exception:
            pass

        deadline = time.monotonic() + _THREAD_STOP_TIMEOUT_SECONDS
        for attr in ("_reader_thread", "_writer_thread"):
            worker = getattr(self, attr)
            if worker is not None and worker.is_alive():
                worker.join(timeout=self._remaining(deadline))
            if worker is None or not worker.is_alive():
                setattr(self, attr, None)
        self._start_ts = None
        if not self.has_live_workers():
            self.discard_output()

    def _validate_output_file(self) -> int:
        if sf is None:  # pragma: no cover - guarded in constructor
            raise NativeRecorderError("Le module soundfile est indisponible.")
        try:
            if not self.output_path.is_file() or self.output_path.stat().st_size <= 0:
                raise NativeRecorderError("Le fichier d'enregistrement est absent ou vide.")
            info = sf.info(str(self.output_path))
        except NativeRecorderError:
            raise
        except Exception as exc:
            raise NativeRecorderError("Le fichier WAV généré est illisible.") from exc

        frames_on_disk = int(getattr(info, "frames", 0) or 0)
        channels_on_disk = int(getattr(info, "channels", 0) or 0)
        samplerate_on_disk = int(getattr(info, "samplerate", 0) or 0)
        if self._frames_written <= 0 or frames_on_disk <= 0:
            raise NativeRecorderError("Aucun échantillon audio n'a été enregistré.")
        if frames_on_disk != self._frames_written:
            raise NativeRecorderError(
                f"Le fichier WAV est incomplet ({frames_on_disk} trames sur {self._frames_written})."
            )
        if channels_on_disk != self._channels:
            raise NativeRecorderError("Le nombre de canaux du fichier WAV est incohérent.")
        if samplerate_on_disk != int(self._samplerate):
            raise NativeRecorderError("La fréquence d'échantillonnage du fichier WAV est incohérente.")
        return frames_on_disk

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def has_live_workers(self) -> bool:
        return any(
            worker is not None and worker.is_alive()
            for worker in (self._reader_thread, self._writer_thread)
        )

    def discard_output(self) -> None:
        try:
            self.output_path.unlink(missing_ok=True)
        except OSError:
            pass


class NativeRecorder:
    """Coordinator object managing native recording sessions."""

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active: Optional[_RecordingSession] = None
        self._completed: Dict[str, NativeRecordingResult] = {}
        self._purge_old_files_locked()

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

            self._purge_old_files_locked()
            device_id, samplerate, channels = self._pick_device()
            recording_id = uuid.uuid4().hex
            output_path = self._base_dir / f"native_recording_{recording_id}.wav"
            session = _RecordingSession(recording_id, output_path, device_id, samplerate, channels)
            self._active = session
            try:
                session.start()
            except Exception:
                if not session.has_live_workers():
                    self._active = None
                    session.discard_output()
                raise
            return NativeRecordingResult(recording_id=recording_id, path=output_path, duration=0.0)

    def stop(self, recording_id: Optional[str] = None) -> NativeRecordingResult:
        """Stop the active recording session."""

        with self._lock:
            if self._active is None:
                raise NativeRecorderNotRunning("Aucun enregistrement natif en cours.")

            session = self._active
            if recording_id and recording_id != session.recording_id:
                raise NativeRecorderNotRunning("Identifiant d'enregistrement invalide.")

            try:
                result = session.stop()
            except Exception:
                if not session.has_live_workers():
                    self._active = None
                    session.discard_output()
                raise
            self._active = None
            self._completed[result.recording_id] = result
            return result

    def get_completed(self, recording_id: str) -> Optional[NativeRecordingResult]:
        with self._lock:
            self._purge_old_files_locked()
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
                return device_id, 48000.0, min(_MAX_CAPTURE_CHANNELS, max(1, channels))

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
        return device_id, 48000.0, min(_MAX_CAPTURE_CHANNELS, max(1, channels))

    def _purge_old_files_locked(self, max_age_seconds: float = 3600.0) -> None:
        now = time.time()
        active_path = self._active.output_path if self._active is not None else None

        # Scan the directory as the source of truth so recordings orphaned by
        # a crash or a previous process are also reclaimed.
        try:
            candidates = list(self._base_dir.iterdir())
        except OSError:  # pragma: no cover - transient filesystem failure
            candidates = []
        for candidate in candidates:
            match = _RECORDING_NAME_RE.fullmatch(candidate.name)
            if match is None or candidate == active_path:
                continue
            try:
                if candidate.is_file() and now - candidate.stat().st_mtime > max_age_seconds:
                    candidate.unlink()
                    self._completed.pop(match.group(1), None)
            except OSError:  # pragma: no cover - best effort cleanup
                continue

        # Reconcile tracked entries even if their file was removed externally
        # or uses a legacy path outside the generated filename convention.
        for rec_id, result in list(self._completed.items()):
            if active_path is not None and result.path == active_path:
                continue
            try:
                if not result.path.exists():
                    self._completed.pop(rec_id, None)
                    continue
                if now - result.path.stat().st_mtime <= max_age_seconds:
                    continue
                result.path.unlink()
            except OSError:  # keep it tracked so a later purge can retry
                continue
            self._completed.pop(rec_id, None)

