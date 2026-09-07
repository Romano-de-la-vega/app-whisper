"""Recording lifecycle, persistent results and recovery, independent of Qt."""

from __future__ import annotations

import queue
import shutil
import threading
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from transcripteur_whisper.audio.native_recorder import NativeRecorder, NativeRecordingResult
from transcripteur_whisper.audio.recovery import recover_wav
from transcripteur_whisper.audio.windows_com import com_apartment
from transcripteur_whisper.models.audio_devices import AudioDeviceSnapshot
from transcripteur_whisper.services.audio_device_service import AudioDeviceError, AudioDeviceService

T = TypeVar("T")


class RecordingService:
    """All stream open/close operations share a dedicated COM apartment thread.

    start/stop/recover are blocking service methods, called by UI workers.
    is_active and levels are cheap state reads and never enumerate devices.
    """

    def __init__(self, paths: Any, devices: AudioDeviceService):
        self.paths = paths
        self.devices = devices
        self._journal_dir = Path(paths.temp) / "recordings"
        self._recorder = NativeRecorder(self._journal_dir)
        self._commands: queue.Queue = queue.Queue()
        self._worker_ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, name="recording-coordinator", daemon=True)
        self._worker.start()
        self._active_result: NativeRecordingResult | None = None
        self._last_completed: NativeRecordingResult | None = None
        self._pending_final: NativeRecordingResult | None = None
        self._resources_active = False
        self._selected: dict[str, str] = {}
        self._closed = False

    def _run(self) -> None:
        try:
            with com_apartment():
                self._worker_ready.set()
                while True:
                    command = self._commands.get()
                    if command is None:
                        return
                    callback, future = command
                    try:
                        future.set_result(callback())
                    except BaseException as exc:
                        future.set_exception(exc)
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._worker_ready.set()

    def _call(self, callback: Callable[[], T]) -> T:
        if self._closed:
            raise RuntimeError("Le service d'enregistrement est fermé.")
        if not self._worker_ready.wait(10):
            raise AudioDeviceError("Le moteur audio ne répond pas à l'initialisation.")
        if self._startup_error:
            raise AudioDeviceError(f"Initialisation du moteur audio impossible : {self._startup_error}")
        future: Future = Future()
        self._commands.put((callback, future))
        return future.result()

    @property
    def is_active(self) -> bool:
        return self._active_result is not None or self._resources_active

    def start(
        self,
        microphone_key: str | None = None,
        speaker_key: str | None = None,
        *,
        microphone_enabled: bool = True,
        system_enabled: bool = True,
    ) -> NativeRecordingResult:
        return self._call(
            lambda: self._start(microphone_key, speaker_key, microphone_enabled, system_enabled)
        )

    def _start(
        self,
        microphone_key: str | None,
        speaker_key: str | None,
        microphone_enabled: bool,
        system_enabled: bool,
    ) -> NativeRecordingResult:
        with self.devices.gate:
            if self._active_result is not None:
                if self._recorder.has_live_capture():
                    return replace(self._active_result, resumed=True)
                # A dead coordinator must not strand the user in a recording
                # that can never deliver samples. Its journals stay on disk.
                self._active_result = None
                self._selected = {}
                self._resources_active = False
                self.devices.set_capture_active(False)
            if not microphone_enabled and not system_enabled:
                raise AudioDeviceError("Activez le microphone ou le son du PC.")
            snapshot = self.devices.snapshot(force_refresh=True)
            microphone = (
                self.devices.resolve(microphone_key, "input", snapshot=snapshot)
                if microphone_enabled
                else None
            )
            speaker = (
                self.devices.resolve(speaker_key, "output", snapshot=snapshot) if system_enabled else None
            )
            self.devices.set_capture_active(True)
            try:
                result = self._recorder.start(
                    f"sd:{microphone.backend_index}" if microphone else None,
                    speaker.backend_id if speaker else None,
                    microphone_enabled=microphone_enabled,
                    system_enabled=system_enabled,
                )
                self._selected = {}
                if microphone:
                    self._selected["microphone"] = microphone.stable_key
                if speaker:
                    self._selected["system"] = speaker.stable_key
                self._active_result = result
                self._last_completed = None
                self._pending_final = None
                return result
            finally:
                self._resources_active = self._recorder.has_live_capture()
                self.devices.set_capture_active(self._resources_active)

    def stop(self) -> NativeRecordingResult:
        return self._call(self._stop)

    def _stop(self) -> NativeRecordingResult:
        with self.devices.gate:
            if not self.is_active and self._last_completed is not None:
                return self._last_completed
            try:
                result = self._pending_final or self._recorder.stop(
                    self._active_result.recording_id if self._active_result else None
                )
                self._pending_final = result
                self.paths.results.mkdir(parents=True, exist_ok=True)
                destination = self.paths.results / result.path.name
                # A failed copy leaves the journal/final WAV recoverable.
                partial = destination.with_suffix(".wav.partial")
                try:
                    shutil.copyfile(result.path, partial)
                    partial.replace(destination)
                finally:
                    partial.unlink(missing_ok=True)
                self._last_completed = NativeRecordingResult(
                    recording_id=result.recording_id,
                    path=destination,
                    duration=result.duration,
                    system_device_name=result.system_device_name,
                    microphone_device_name=result.microphone_device_name,
                )
                self._pending_final = None
                # Cleanup failure does not invalidate a safely persisted WAV.
                try:
                    result.path.unlink(missing_ok=True)
                except OSError:
                    pass
                return self._last_completed
            finally:
                live = self._recorder.has_live_capture()
                self._resources_active = live
                self.devices.set_capture_active(live)
                if not live:
                    self._active_result = None
                    self._selected = {}

    def levels(self) -> dict | None:
        active = self._active_result
        return self._recorder.get_levels(active.recording_id) if active else None

    def device_snapshot_changed(self, snapshot: AudioDeviceSnapshot) -> None:
        """Never substitute a different device into an already open stream."""
        with self.devices.gate:
            # A start can refresh the devices after a monitor scan but before
            # this callback. Applying that old scan to the newly opened source
            # would falsely declare it disconnected. Start/stop and this check
            # share the same gate; no COM object is closed by this callback.
            if snapshot is not self.devices.current_snapshot:
                return
            keys = {item.stable_key for item in snapshot.all}
            for source, key in tuple(self._selected.items()):
                if key not in keys:
                    self._recorder.source_unavailable(source)

    def recoverable_recordings(self) -> tuple[Path, ...]:
        active_id = self._active_result.recording_id if self._active_result else None
        return tuple(
            sorted(
                path
                for path in self._journal_dir.glob("*.wav")
                if (path.name.startswith(".native_recording_") or path.name.startswith("native_recording_"))
                and (not active_id or active_id not in path.name)
            )
        )

    def recover(self, path: Path) -> Path:
        source = Path(path).resolve()
        if source.parent != self._journal_dir.resolve() or source not in self.recoverable_recordings():
            raise AudioDeviceError("Ce fichier ne fait pas partie des enregistrements récupérables.")
        return recover_wav(source, self.paths.results)

    def close(self) -> None:
        """Call from a worker; a stuck capture raises and permits a stop retry."""
        if not self._closed:
            if self.is_active:
                self.stop()
            self._closed = True
            self._commands.put(None)
        self._worker.join(timeout=15)
        if self._worker.is_alive():
            raise AudioDeviceError("Le moteur audio n'a pas terminé son arrêt.")
