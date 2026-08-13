"""Windows native recorder: system loopback with SoundCard + microphone with sounddevice.

Designed for the Transcripteur Whisper application.

Why two backends?
- SoundCard is convenient for Windows WASAPI loopback (PC/system audio).
- python-sounddevice/PortAudio is used for physical microphones, avoiding
  SoundCard's known Windows/WASAPI single-channel recording issue.

Public API kept compatible with server.py:
    NativeRecorder(base_dir)
    .is_available()
    .list_devices()
    .start(microphone_id=None, speaker_id=None)
    .get_levels(recording_id)
    .stop(recording_id=None)
    .get_completed(recording_id)
"""

from __future__ import annotations

import math
import os
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


_OUTPUT_CHANNELS = 2
_SYSTEM_CAPTURE_FRAMES = 2048
_SYSTEM_BLOCKSIZE = 8192
_QUEUE_MAX_BLOCKS = 512
_QUEUE_PUT_TIMEOUT = 0.1
_STOP_TIMEOUT = 10.0
_READER_GRACE = 0.75
_MIX_CHUNK_FRAMES = 65536
_METER_FLOOR_DB = -60.0
_METER_STALE_SECONDS = 0.8
_RECORDING_RE = re.compile(r"native_recording_([0-9a-f]{32})\.wav\Z")

_SYSTEM_GAIN = max(0.0, float(os.getenv("WHISPER_SYSTEM_AUDIO_GAIN", "0.80")))
_MIC_GAIN = max(0.0, float(os.getenv("WHISPER_MICROPHONE_GAIN", "1.00")))

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None  # type: ignore

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore


class NativeRecorderError(RuntimeError):
    pass


class NativeRecorderUnsupported(NativeRecorderError):
    pass


class NativeRecorderBusy(NativeRecorderError):
    pass


class NativeRecorderNotRunning(NativeRecorderError):
    pass


@dataclass(frozen=True)
class NativeRecordingResult:
    recording_id: str
    path: Path
    duration: float
    system_device_name: str = ""
    microphone_device_name: str = ""


@dataclass
class _SourceState:
    key: str
    path: Path
    gain: float
    audio_queue: "queue.Queue[Any]" = field(
        default_factory=lambda: queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
    )
    reader_done: threading.Event = field(default_factory=threading.Event)
    writer_ready: threading.Event = field(default_factory=threading.Event)
    writer_done: threading.Event = field(default_factory=threading.Event)
    writer_thread: Optional[threading.Thread] = None
    frames_written: int = 0
    first_frame_monotonic: Optional[float] = None
    last_frame_monotonic: Optional[float] = None


class _RecordingSession:
    def __init__(
        self,
        recording_id: str,
        output_path: Path,
        microphone_index: int,
        microphone_name: str,
        speaker_id: str,
        speaker_name: str,
    ) -> None:
        if np is None or sf is None or sc is None or sd is None:
            raise NativeRecorderUnsupported(
                "Les modules numpy, soundfile, SoundCard et sounddevice sont requis."
            )

        self.recording_id = recording_id
        self.output_path = output_path
        self.microphone_index = int(microphone_index)
        self.microphone_name = microphone_name
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name

        self._stop_event = threading.Event()
        self._closing_system = threading.Event()
        self._state_lock = threading.RLock()
        self._error_lock = threading.Lock()
        self._worker_failure: Optional[tuple[str, BaseException]] = None

        self._samplerate = 48000
        self._mic_channels = 1
        self._start_ts: Optional[float] = None
        self._stop_ts: Optional[float] = None

        self._mic_stream: Optional[Any] = None
        self._system_context: Optional[Any] = None
        self._system_recorder: Optional[Any] = None
        self._system_reader_thread: Optional[threading.Thread] = None

        self._sources: Dict[str, _SourceState] = {
            "system": _SourceState(
                key="system",
                path=output_path.with_name(f".{output_path.stem}_system.wav"),
                gain=_SYSTEM_GAIN,
            ),
            "microphone": _SourceState(
                key="microphone",
                path=output_path.with_name(f".{output_path.stem}_microphone.wav"),
                gain=_MIC_GAIN,
            ),
        }

        self._levels: Dict[str, Dict[str, Any]] = {
            "system": {
                "rms_db": _METER_FLOOR_DB,
                "peak_db": _METER_FLOOR_DB,
                "clipped": False,
                "updated_monotonic": 0.0,
            },
            "microphone": {
                "rms_db": _METER_FLOOR_DB,
                "peak_db": _METER_FLOOR_DB,
                "clipped": False,
                "updated_monotonic": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.unlink(missing_ok=True)
        for source in self._sources.values():
            source.path.unlink(missing_ok=True)

        mic_info = sd.query_devices(self.microphone_index, "input")
        default_rate = int(round(float(mic_info.get("default_samplerate") or 0)))
        max_input_channels = int(mic_info.get("max_input_channels") or 0)
        if max_input_channels <= 0:
            raise NativeRecorderUnsupported(
                f"Le périphérique « {self.microphone_name} » n'a aucun canal d'entrée."
            )

        rate_candidates: list[int] = []
        for rate in (default_rate, 48000, 44100):
            if rate > 0 and rate not in rate_candidates:
                rate_candidates.append(rate)

        channel_candidates = [1]
        if max_input_channels >= 2:
            channel_candidates.append(2)

        attempts: list[str] = []
        last_exc: Optional[BaseException] = None

        # We open the microphone with sounddevice and system loopback with
        # SoundCard at the same sample rate so no resampler is needed later.
        for rate in rate_candidates:
            for mic_channels in channel_candidates:
                self._reset_runtime_state()
                try:
                    self._samplerate = rate
                    self._mic_channels = mic_channels
                    self._open_microphone_stream()
                    self._open_system_loopback()
                    self._start_writer_threads()
                    self._start_ts = time.monotonic()
                    self._start_system_reader()
                    self._mic_stream.start()
                    self._raise_worker_failure()
                    return
                except Exception as exc:
                    last_exc = exc
                    attempts.append(
                        f"{rate} Hz, micro={mic_channels} canal(aux): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._cleanup_failed_attempt()

        recent = " | ".join(attempts[-6:])
        detail = f" Dernière erreur : {type(last_exc).__name__}: {last_exc}." if last_exc else ""
        raise NativeRecorderUnsupported(
            f"Impossible d'ouvrir le microphone « {self.microphone_name} » avec sounddevice/PortAudio."
            f"{detail} Essais : {recent}"
        ) from last_exc

    def stop(self) -> NativeRecordingResult:
        if self._start_ts is None:
            raise NativeRecorderNotRunning("Aucun enregistrement natif en cours.")

        self._stop_ts = time.monotonic()
        self._stop_event.set()
        deadline = time.monotonic() + _STOP_TIMEOUT

        self._close_microphone_stream()
        self._sources["microphone"].reader_done.set()

        reader = self._system_reader_thread
        if reader is not None and reader.is_alive():
            reader.join(timeout=min(_READER_GRACE, self._remaining(deadline)))

        self._close_system_loopback()
        if reader is not None and reader.is_alive():
            reader.join(timeout=self._remaining(deadline))
        self._sources["system"].reader_done.set()

        stuck: list[str] = []
        if reader is not None and reader.is_alive():
            stuck.append("capture système")

        for key, source in self._sources.items():
            writer = source.writer_thread
            if writer is not None and writer.is_alive():
                writer.join(timeout=self._remaining(deadline))
            if writer is not None and writer.is_alive():
                stuck.append(f"écriture {key}")

        if stuck:
            raise NativeRecorderError(
                "Arrêt incomplet : threads toujours actifs (" + ", ".join(stuck) + ")."
            )

        self._raise_worker_failure()
        self._validate_sources()

        try:
            duration = self._mix_sources()
        finally:
            self._discard_source_files()

        self._start_ts = None
        return NativeRecordingResult(
            recording_id=self.recording_id,
            path=self.output_path,
            duration=duration,
            system_device_name=self.speaker_name,
            microphone_device_name=self.microphone_name,
        )

    def has_live_workers(self) -> bool:
        if self._system_reader_thread is not None and self._system_reader_thread.is_alive():
            return True
        for source in self._sources.values():
            if source.writer_thread is not None and source.writer_thread.is_alive():
                return True
        return False

    def discard_output(self) -> None:
        self.output_path.unlink(missing_ok=True)
        self._discard_source_files()

    # ------------------------------------------------------------------
    # Device opening
    # ------------------------------------------------------------------
    def _open_microphone_stream(self) -> None:
        try:
            sd.check_input_settings(
                device=self.microphone_index,
                channels=self._mic_channels,
                samplerate=self._samplerate,
                dtype="float32",
            )
            self._mic_stream = sd.InputStream(
                device=self.microphone_index,
                samplerate=self._samplerate,
                channels=self._mic_channels,
                dtype="float32",
                blocksize=0,
                latency="high",
                callback=self._microphone_callback,
            )
        except Exception as exc:
            raise NativeRecorderUnsupported(
                f"PortAudio refuse « {self.microphone_name} » "
                f"({self._samplerate} Hz, {self._mic_channels} canal(aux)) : {exc}"
            ) from exc

    def _open_system_loopback(self) -> None:
        try:
            try:
                speaker = sc.get_speaker(self.speaker_id)
            except Exception:
                speaker = sc.get_speaker(self.speaker_name)
            loopback = sc.get_microphone(
                id=getattr(speaker, "id", self.speaker_id),
                include_loopback=True,
            )
            channels = _device_channel_count(loopback)
            if channels <= 0:
                raise NativeRecorderUnsupported(
                    f"La sortie « {self.speaker_name} » n'expose aucun canal loopback."
                )
            channel_map = list(range(min(_OUTPUT_CHANNELS, channels)))
            self._system_context = loopback.recorder(
                samplerate=self._samplerate,
                channels=channel_map,
                blocksize=_SYSTEM_BLOCKSIZE,
                exclusive_mode=False,
            )
            self._system_recorder = self._system_context.__enter__()
        except Exception as exc:
            raise NativeRecorderUnsupported(
                f"Impossible d'ouvrir le loopback du son PC « {self.speaker_name} » "
                f"à {self._samplerate} Hz : {exc}"
            ) from exc

    def _close_microphone_stream(self) -> None:
        stream = self._mic_stream
        self._mic_stream = None
        if stream is None:
            return
        try:
            if getattr(stream, "active", False):
                stream.stop()
        finally:
            stream.close()

    def _close_system_loopback(self) -> None:
        context = self._system_context
        self._system_context = None
        self._system_recorder = None
        if context is None:
            return
        self._closing_system.set()
        try:
            context.__exit__(None, None, None)
        except Exception:
            # Closing the context is also how a blocked SoundCard record() call
            # is interrupted during shutdown.
            pass

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _microphone_callback(self, indata, frames, time_info, status) -> None:
        if self._stop_event.is_set():
            return
        try:
            data = np.asarray(indata, dtype="float32").copy()
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2 or data.shape[0] <= 0:
                return

            # Voice is reduced to mono then duplicated to stereo. This also
            # handles WASAPI endpoints that advertise 2 input channels for a
            # headset microphone.
            mono = np.mean(data, axis=1, dtype="float32")
            stereo = np.column_stack((mono, mono)).astype("float32", copy=False)

            now = time.monotonic()
            source = self._sources["microphone"]
            estimated_start = now - (stereo.shape[0] / self._samplerate)
            with self._state_lock:
                if source.first_frame_monotonic is None:
                    source.first_frame_monotonic = estimated_start
                source.last_frame_monotonic = now

            self._update_meter("microphone", stereo * source.gain)
            try:
                source.audio_queue.put_nowait(stereo)
            except queue.Full as exc:
                self._record_failure(
                    "capture microphone",
                    NativeRecorderError("La file audio microphone est saturée."),
                )
                raise sd.CallbackAbort from exc
        except sd.CallbackAbort:
            raise
        except Exception as exc:
            self._record_failure("capture microphone", exc)
            raise sd.CallbackAbort from exc

    def _start_system_reader(self) -> None:
        self._system_reader_thread = threading.Thread(
            target=self._system_reader_loop,
            daemon=True,
            name="native-system-loopback-reader",
        )
        self._system_reader_thread.start()

    def _system_reader_loop(self) -> None:
        source = self._sources["system"]
        try:
            while not self._stop_event.is_set():
                recorder = self._system_recorder
                if recorder is None:
                    break
                chunk = recorder.record(numframes=_SYSTEM_CAPTURE_FRAMES)
                now = time.monotonic()
                if chunk is None or getattr(chunk, "size", 0) == 0:
                    continue
                data = self._normalise_system_stereo(chunk)
                estimated_start = now - (data.shape[0] / self._samplerate)
                with self._state_lock:
                    if source.first_frame_monotonic is None:
                        source.first_frame_monotonic = estimated_start
                    source.last_frame_monotonic = now
                self._update_meter("system", data * source.gain)
                while not self._stop_event.is_set():
                    try:
                        source.audio_queue.put(data, timeout=_QUEUE_PUT_TIMEOUT)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            if not self._closing_system.is_set() and not self._stop_event.is_set():
                self._record_failure("capture système", exc)
        finally:
            source.reader_done.set()

    def _normalise_system_stereo(self, chunk: Any) -> Any:
        data = np.asarray(chunk, dtype="float32")
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
            raise NativeRecorderError(f"Bloc loopback invalide : {data.shape}.")
        if data.shape[1] == 1:
            return np.repeat(data, 2, axis=1)
        return data[:, :2]

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------
    def _start_writer_threads(self) -> None:
        for key, source in self._sources.items():
            source.writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(key,),
                daemon=True,
                name=f"native-audio-writer-{key}",
            )
            source.writer_thread.start()
        for source in self._sources.values():
            if not source.writer_ready.wait(timeout=_STOP_TIMEOUT):
                raise NativeRecorderError(
                    f"Le writer audio « {source.key} » ne s'est pas initialisé."
                )
        self._raise_worker_failure()

    def _writer_loop(self, key: str) -> None:
        source = self._sources[key]
        try:
            with sf.SoundFile(
                source.path,
                mode="w",
                samplerate=self._samplerate,
                channels=_OUTPUT_CHANNELS,
                subtype="PCM_16",
            ) as output_file:
                source.writer_ready.set()
                while True:
                    try:
                        data = source.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        if source.reader_done.is_set():
                            break
                        continue
                    try:
                        array = np.asarray(data, dtype="float32")
                        if array.ndim != 2 or array.shape[1] != 2:
                            raise NativeRecorderError(
                                f"Bloc {key} inattendu : {array.shape}."
                            )
                        output_file.write(np.clip(array, -1.0, 1.0))
                        with self._state_lock:
                            source.frames_written += int(array.shape[0])
                    finally:
                        source.audio_queue.task_done()
        except Exception as exc:
            self._record_failure(f"écriture {key}", exc)
        finally:
            source.writer_ready.set()
            source.writer_done.set()

    # ------------------------------------------------------------------
    # VU meters
    # ------------------------------------------------------------------
    def _update_meter(self, key: str, data: Any) -> None:
        absolute = np.abs(np.asarray(data, dtype="float32"))
        if absolute.size == 0:
            return
        peak = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(np.square(absolute, dtype="float64"))))
        peak_db = self._to_db(peak)
        rms_db = self._to_db(rms)
        with self._state_lock:
            previous = self._levels[key]
            if previous["updated_monotonic"]:
                peak_db = max(peak_db, float(previous["peak_db"]) - 3.0)
                rms_db = 0.35 * rms_db + 0.65 * float(previous["rms_db"])
            self._levels[key] = {
                "rms_db": max(_METER_FLOOR_DB, rms_db),
                "peak_db": max(_METER_FLOOR_DB, peak_db),
                "clipped": peak >= 0.999,
                "updated_monotonic": time.monotonic(),
            }

    @staticmethod
    def _to_db(value: float) -> float:
        if value <= 0:
            return _METER_FLOOR_DB
        return max(_METER_FLOOR_DB, 20.0 * math.log10(value))

    def levels_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            sources: Dict[str, Any] = {}
            for key, source in self._sources.items():
                meter = dict(self._levels[key])
                updated = float(meter["updated_monotonic"] or 0.0)
                age = (now - updated) if updated else None
                stale = age is None or age > _METER_STALE_SECONDS
                peak = _METER_FLOOR_DB if stale else float(meter["peak_db"])
                rms = _METER_FLOOR_DB if stale else float(meter["rms_db"])
                sources[key] = {
                    "name": self.speaker_name if key == "system" else self.microphone_name,
                    "gain": source.gain,
                    "rms_db": round(rms, 1),
                    "peak_db": round(peak, 1),
                    "clipped": False if stale else bool(meter["clipped"]),
                    "signal": peak > -50.0,
                    "age_ms": None if age is None else int(max(0.0, age) * 1000),
                    "frames_captured": int(source.frames_written),
                }
            elapsed = 0.0
            if self._start_ts is not None:
                elapsed = max(0.0, (self._stop_ts or now) - self._start_ts)
            return {
                "recording_id": self.recording_id,
                "active": not self._stop_event.is_set(),
                "sample_rate": self._samplerate,
                "elapsed": elapsed,
                "sources": sources,
            }

    # ------------------------------------------------------------------
    # Final mix
    # ------------------------------------------------------------------
    def _validate_sources(self) -> None:
        for key, source in self._sources.items():
            try:
                info = sf.info(str(source.path))
            except Exception as exc:
                raise NativeRecorderError(
                    f"Le fichier temporaire {key} est illisible."
                ) from exc
            if int(info.frames or 0) <= 0:
                raise NativeRecorderError(
                    f"Aucun échantillon n'a été enregistré pour {key}."
                )
            if int(info.samplerate or 0) != self._samplerate or int(info.channels or 0) != 2:
                raise NativeRecorderError(f"Format temporaire incohérent pour {key}.")

    def _mix_sources(self) -> float:
        starts = {
            key: source.first_frame_monotonic
            for key, source in self._sources.items()
        }
        valid_starts = [value for value in starts.values() if value is not None]
        if not valid_starts:
            raise NativeRecorderError("Impossible d'aligner les sources audio.")
        global_start = min(valid_starts)

        details: Dict[str, Dict[str, Any]] = {}
        total_frames = 0
        for key, source in self._sources.items():
            info = sf.info(str(source.path))
            source_start = starts[key] if starts[key] is not None else global_start
            offset = max(0, int(round((source_start - global_start) * self._samplerate)))
            frames = int(info.frames)
            details[key] = {"source": source, "offset": offset, "frames": frames}
            total_frames = max(total_frames, offset + frames)

        opened: Dict[str, Any] = {}
        self.output_path.unlink(missing_ok=True)
        try:
            for key, detail in details.items():
                opened[key] = sf.SoundFile(detail["source"].path, mode="r")

            with sf.SoundFile(
                self.output_path,
                mode="w",
                samplerate=self._samplerate,
                channels=2,
                subtype="PCM_16",
            ) as output:
                for output_start in range(0, total_frames, _MIX_CHUNK_FRAMES):
                    count = min(_MIX_CHUNK_FRAMES, total_frames - output_start)
                    blocks: Dict[str, Any] = {}
                    for key, detail in details.items():
                        block = np.zeros((count, 2), dtype="float32")
                        source_start = int(detail["offset"])
                        source_end = source_start + int(detail["frames"])
                        overlap_start = max(output_start, source_start)
                        overlap_end = min(output_start + count, source_end)
                        if overlap_end > overlap_start:
                            handle = opened[key]
                            handle.seek(overlap_start - source_start)
                            data = handle.read(
                                overlap_end - overlap_start,
                                dtype="float32",
                                always_2d=True,
                            )
                            destination = overlap_start - output_start
                            block[destination : destination + data.shape[0], :] = data[:, :2]
                        blocks[key] = block

                    mixed = (
                        blocks["system"] * self._sources["system"].gain
                        + blocks["microphone"] * self._sources["microphone"].gain
                    )
                    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
                    if peak > 0.999:
                        mixed *= 0.999 / peak
                    output.write(np.clip(mixed, -1.0, 1.0))
        except Exception:
            self.output_path.unlink(missing_ok=True)
            raise
        finally:
            for handle in opened.values():
                try:
                    handle.close()
                except Exception:
                    pass

        info = sf.info(str(self.output_path))
        return int(info.frames) / max(self._samplerate, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_runtime_state(self) -> None:
        self._stop_event.clear()
        self._closing_system.clear()
        self._worker_failure = None
        self._system_reader_thread = None
        self._mic_stream = None
        self._system_context = None
        self._system_recorder = None
        for source in self._sources.values():
            source.audio_queue = queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
            source.reader_done.clear()
            source.writer_ready.clear()
            source.writer_done.clear()
            source.writer_thread = None
            source.frames_written = 0
            source.first_frame_monotonic = None
            source.last_frame_monotonic = None
            source.path.unlink(missing_ok=True)

    def _cleanup_failed_attempt(self) -> None:
        self._stop_event.set()
        try:
            self._close_microphone_stream()
        except Exception:
            pass
        try:
            self._close_system_loopback()
        except Exception:
            pass
        self._sources["microphone"].reader_done.set()
        self._sources["system"].reader_done.set()
        deadline = time.monotonic() + 2.0
        if self._system_reader_thread is not None and self._system_reader_thread.is_alive():
            self._system_reader_thread.join(timeout=self._remaining(deadline))
        for source in self._sources.values():
            if source.writer_thread is not None and source.writer_thread.is_alive():
                source.writer_thread.join(timeout=self._remaining(deadline))
        self._discard_source_files()
        self._start_ts = None

    def _discard_source_files(self) -> None:
        for source in self._sources.values():
            try:
                source.path.unlink(missing_ok=True)
            except OSError:
                pass

    def _record_failure(self, role: str, exc: BaseException) -> None:
        with self._error_lock:
            if self._worker_failure is None:
                self._worker_failure = (role, exc)
        self._stop_event.set()

    def _raise_worker_failure(self) -> None:
        with self._error_lock:
            failure = self._worker_failure
        if failure is not None:
            role, exc = failure
            raise NativeRecorderError(f"Erreur du thread de {role} audio : {exc}") from exc

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())


class NativeRecorder:
    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active: Optional[_RecordingSession] = None
        self._completed: Dict[str, NativeRecordingResult] = {}
        self._purge_old_files_locked()

    def is_available(self) -> bool:
        if sys.platform != "win32" or np is None or sf is None or sc is None or sd is None:
            return False
        try:
            return bool(self._wasapi_microphones()) and sc.default_speaker() is not None
        except Exception:
            return False

    def list_devices(self) -> Dict[str, Any]:
        self._require_backend()
        microphones = self._wasapi_microphones()
        if not microphones:
            raise NativeRecorderUnsupported(
                "Aucun microphone Windows WASAPI n'a été détecté par sounddevice."
            )

        try:
            speakers = list(sc.all_speakers())
            default_speaker = sc.default_speaker()
        except Exception as exc:
            raise NativeRecorderUnsupported(
                "Impossible de lister les sorties audio Windows avec SoundCard."
            ) from exc

        default_speaker_id = str(getattr(default_speaker, "id", ""))
        speaker_payload = []
        for speaker in speakers:
            speaker_payload.append(
                {
                    "id": str(getattr(speaker, "id", "")),
                    "name": str(getattr(speaker, "name", "Sortie audio")),
                    "channels": _device_channel_count(speaker),
                    "default": str(getattr(speaker, "id", "")) == default_speaker_id,
                }
            )

        return {"microphones": microphones, "speakers": speaker_payload}

    def start(
        self,
        microphone_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
    ) -> NativeRecordingResult:
        with self._lock:
            if self._active is not None:
                raise NativeRecorderBusy("Un enregistrement est déjà en cours.")

            self._purge_old_files_locked()
            mic_index, mic_name = self._resolve_microphone(microphone_id)
            speaker_id_resolved, speaker_name = self._resolve_speaker(speaker_id)

            recording_id = uuid.uuid4().hex
            output_path = self._base_dir / f"native_recording_{recording_id}.wav"
            session = _RecordingSession(
                recording_id=recording_id,
                output_path=output_path,
                microphone_index=mic_index,
                microphone_name=mic_name,
                speaker_id=speaker_id_resolved,
                speaker_name=speaker_name,
            )
            self._active = session

        try:
            session.start()
        except Exception:
            with self._lock:
                if self._active is session and not session.has_live_workers():
                    self._active = None
            session.discard_output()
            raise

        return NativeRecordingResult(
            recording_id=recording_id,
            path=output_path,
            duration=0.0,
            system_device_name=speaker_name,
            microphone_device_name=mic_name,
        )

    def stop(self, recording_id: Optional[str] = None) -> NativeRecordingResult:
        with self._lock:
            session = self._active
            if session is None:
                raise NativeRecorderNotRunning("Aucun enregistrement natif en cours.")
            if recording_id and recording_id != session.recording_id:
                raise NativeRecorderNotRunning("Identifiant d'enregistrement invalide.")

        try:
            result = session.stop()
        except Exception:
            with self._lock:
                if self._active is session and not session.has_live_workers():
                    self._active = None
            session.discard_output()
            raise

        with self._lock:
            if self._active is session:
                self._active = None
            self._completed[result.recording_id] = result
        return result

    def get_levels(self, recording_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._active
        if session is None or session.recording_id != recording_id:
            return None
        return session.levels_snapshot()

    def get_completed(self, recording_id: str) -> Optional[NativeRecordingResult]:
        with self._lock:
            self._purge_old_files_locked()
            return self._completed.get(recording_id)

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------
    def _require_backend(self) -> None:
        if sys.platform != "win32":
            raise NativeRecorderUnsupported("L'enregistrement natif est disponible sous Windows.")
        missing = []
        if np is None:
            missing.append("numpy")
        if sf is None:
            missing.append("soundfile")
        if sc is None:
            missing.append("SoundCard")
        if sd is None:
            missing.append("sounddevice")
        if missing:
            raise NativeRecorderUnsupported(
                "Dépendances audio manquantes : " + ", ".join(missing) + "."
            )

    def _wasapi_microphones(self) -> list[Dict[str, Any]]:
        self._require_backend()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        default_name = ""
        try:
            default_input = sd.query_devices(kind="input")
            default_name = str(default_input.get("name") or "")
        except Exception:
            pass

        payload: list[Dict[str, Any]] = []
        for index, device in enumerate(devices):
            max_inputs = int(device.get("max_input_channels") or 0)
            if max_inputs <= 0:
                continue
            hostapi_index = int(device.get("hostapi") or 0)
            try:
                hostapi_name = str(hostapis[hostapi_index].get("name") or "")
            except Exception:
                hostapi_name = ""
            if "WASAPI" not in hostapi_name.upper():
                continue
            name = str(device.get("name") or f"Microphone {index}")
            payload.append(
                {
                    "id": f"sd:{index}",
                    "name": name,
                    "channels": max_inputs,
                    "sample_rate": int(round(float(device.get("default_samplerate") or 0))),
                    "host_api": hostapi_name,
                    "default": bool(default_name and _same_device_name(name, default_name)),
                }
            )

        if payload and not any(item["default"] for item in payload):
            payload[0]["default"] = True
        return payload

    def _resolve_microphone(self, microphone_id: Optional[str]) -> tuple[int, str]:
        microphones = self._wasapi_microphones()
        if not microphones:
            raise NativeRecorderUnsupported("Aucun microphone WASAPI disponible.")

        if microphone_id:
            token = str(microphone_id).strip()
            if token.startswith("sd:"):
                token = token[3:]
            try:
                wanted = int(token)
            except ValueError:
                wanted = -1
            for mic in microphones:
                if int(str(mic["id"]).split(":", 1)[1]) == wanted:
                    return wanted, str(mic["name"])
            raise NativeRecorderUnsupported(
                "Le microphone sélectionné n'est plus disponible. Actualisez la liste."
            )

        selected = next((mic for mic in microphones if mic.get("default")), microphones[0])
        index = int(str(selected["id"]).split(":", 1)[1])
        return index, str(selected["name"])

    def _resolve_speaker(self, speaker_id: Optional[str]) -> tuple[str, str]:
        try:
            if speaker_id:
                try:
                    speaker = sc.get_speaker(str(speaker_id))
                except Exception:
                    speakers = list(sc.all_speakers())
                    speaker = next(
                        (item for item in speakers if str(getattr(item, "id", "")) == str(speaker_id)),
                        None,
                    )
                    if speaker is None:
                        raise NativeRecorderUnsupported(
                            "La sortie audio sélectionnée n'est plus disponible. Actualisez la liste."
                        )
            else:
                speaker = sc.default_speaker()
            if speaker is None:
                raise NativeRecorderUnsupported("Aucune sortie audio Windows disponible.")
            return str(getattr(speaker, "id", "")), str(getattr(speaker, "name", "Son du PC"))
        except NativeRecorderUnsupported:
            raise
        except Exception as exc:
            raise NativeRecorderUnsupported(
                "Impossible de sélectionner la sortie audio Windows."
            ) from exc

    def _purge_old_files_locked(self, max_age_seconds: float = 3600.0) -> None:
        now = time.time()
        active_path = self._active.output_path if self._active is not None else None
        try:
            files = list(self._base_dir.iterdir())
        except OSError:
            files = []
        for path in files:
            if path == active_path:
                continue
            match = _RECORDING_RE.fullmatch(path.name)
            temp = path.name.startswith(".native_recording_") and path.suffix == ".wav"
            if match is None and not temp:
                continue
            try:
                if path.is_file() and now - path.stat().st_mtime > max_age_seconds:
                    path.unlink()
                    if match:
                        self._completed.pop(match.group(1), None)
            except OSError:
                pass


def _device_channel_count(device: Any) -> int:
    raw = getattr(device, "channels", 0)
    if isinstance(raw, int):
        return raw
    try:
        return len(raw)
    except Exception:
        return 0


def _normalise_name(value: str) -> str:
    text = str(value or "").casefold()
    for token in ("windows wasapi", "windows directsound", "mme", "wdm-ks"):
        text = text.replace(token, "")
    return " ".join(text.replace("®", "").split())


def _same_device_name(left: str, right: str) -> bool:
    a = _normalise_name(left)
    b = _normalise_name(right)
    return bool(a and b and (a == b or a in b or b in a))
