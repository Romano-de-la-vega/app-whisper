import os
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import audio_fix_v2.native_recorder as recorder_module


class _StuckWorker:
    def __init__(self):
        self.join_calls = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls += 1


class NativeRecordingSessionTests(unittest.TestCase):
    def setUp(self):
        if recorder_module.np is None or recorder_module.sf is None:
            self.skipTest("numpy et soundfile sont nécessaires à ces tests")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)

    def _session(self, *, samplerate=8_000, channels=2):
        output = self.base_dir / f"native_recording_{uuid.uuid4().hex}.wav"
        with mock.patch.object(recorder_module, "sc", object()):
            return recorder_module._RecordingSession(
                "recording-id",
                output,
                "device-id",
                samplerate,
                channels,
            )

    def test_queue_is_bounded_and_channels_are_limited_to_stereo(self):
        session = self._session(channels=8)

        self.assertEqual(session._queue.maxsize, recorder_module._QUEUE_MAX_BLOCKS)
        self.assertEqual(session._channels, 2)

    def test_reader_applies_backpressure_and_keeps_captured_block_on_stop(self):
        session = self._session(channels=1)
        sample = recorder_module.np.ones((8,), dtype="float32")
        for _ in range(recorder_module._QUEUE_MAX_BLOCKS):
            session._queue.put_nowait(sample)

        first_record_done = threading.Event()

        class OneBlockRecorder:
            calls = 0

            def record(self, _blocksize):
                self.calls += 1
                first_record_done.set()
                return sample

        fake_recorder = OneBlockRecorder()
        session._recorder = fake_recorder
        reader = threading.Thread(target=session._reader_loop)
        reader.start()
        self.assertTrue(first_record_done.wait(1.0))

        # The queue stays bounded and the reader cannot request another block.
        time.sleep(recorder_module._QUEUE_PUT_TIMEOUT_SECONDS * 1.5)
        self.assertTrue(reader.is_alive())
        self.assertEqual(session._queue.qsize(), recorder_module._QUEUE_MAX_BLOCKS)
        self.assertEqual(fake_recorder.calls, 1)

        # stop_requested must not discard the block already returned by record().
        session._stop_event.set()
        session._queue.get_nowait()
        reader.join(1.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(session._queue.qsize(), recorder_module._QUEUE_MAX_BLOCKS)
        self.assertEqual(fake_recorder.calls, 1)

    def test_stop_drains_block_returned_when_context_is_closed(self):
        session = self._session(samplerate=8_000, channels=2)
        entered_record = threading.Event()
        release_record = threading.Event()
        block = recorder_module.np.full((128, 2), 0.25, dtype="float32")

        class BlockingRecorder:
            def record(self, _blocksize):
                entered_record.set()
                if not release_record.wait(2.0):
                    raise RuntimeError("test timeout")
                return block

        class ReleasingContext:
            def __exit__(self, _exc_type, _exc, _tb):
                release_record.set()

        session._recorder = BlockingRecorder()
        session._recorder_ctx = ReleasingContext()
        session._writer_thread = threading.Thread(target=session._writer_loop)
        session._reader_thread = threading.Thread(target=session._reader_loop)
        session._writer_thread.start()
        self.assertTrue(session._writer_ready_event.wait(1.0))
        session._reader_thread.start()
        self.assertTrue(entered_record.wait(1.0))

        with mock.patch.object(recorder_module, "_READER_STOP_GRACE_SECONDS", 0.01):
            result = session.stop()

        self.assertEqual(session._frames_written, 128)
        self.assertAlmostEqual(result.duration, 128 / 8_000)
        self.assertEqual(recorder_module.sf.info(str(result.path)).frames, 128)

    def test_duration_counts_frames_not_stereo_scalar_samples(self):
        session = self._session(samplerate=48_000, channels=2)
        session._queue.put(recorder_module.np.zeros((480, 2), dtype="float32"))
        session._queue.put(recorder_module.np.zeros((960, 2), dtype="float32"))
        session._reader_done_event.set()

        session._writer_loop()
        result = session.stop()

        self.assertEqual(session._frames_written, 1_440)
        self.assertAlmostEqual(result.duration, 0.03, places=6)

    def test_reader_error_is_chained_from_stop(self):
        session = self._session(channels=1)
        original = RuntimeError("capture cassée")

        class BrokenRecorder:
            def record(self, _blocksize):
                raise original

        session._recorder = BrokenRecorder()
        session._reader_loop()

        with self.assertRaises(recorder_module.NativeRecorderError) as caught:
            session.stop()
        self.assertIs(caught.exception.__cause__, original)
        self.assertIn("capture", str(caught.exception))

    def test_writer_error_is_chained_from_stop(self):
        session = self._session(channels=1)
        original = OSError("disque plein")

        class BrokenSoundFile:
            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                return None

            def write(self, _data):
                raise original

        fake_soundfile_module = mock.Mock()
        fake_soundfile_module.SoundFile.return_value = BrokenSoundFile()
        session._queue.put(recorder_module.np.ones((16,), dtype="float32"))
        session._reader_done_event.set()

        with mock.patch.object(recorder_module, "sf", fake_soundfile_module):
            session._writer_loop()
            with self.assertRaises(recorder_module.NativeRecorderError) as caught:
                session.stop()

        self.assertIs(caught.exception.__cause__, original)
        self.assertIn("écriture", str(caught.exception))

    def test_writer_failure_unblocks_reader_waiting_on_full_queue(self):
        session = self._session(channels=1)
        sample = recorder_module.np.ones((16,), dtype="float32")
        for _ in range(recorder_module._QUEUE_MAX_BLOCKS):
            session._queue.put_nowait(sample)

        reader_started = threading.Event()
        writer_started = threading.Event()
        release_writer = threading.Event()
        original = OSError("disque plein")

        class Recorder:
            def record(self, _blocksize):
                reader_started.set()
                return sample

        class BrokenSoundFile:
            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                return None

            def write(self, _data):
                writer_started.set()
                if not release_writer.wait(1.0):
                    raise RuntimeError("test timeout")
                raise original

        fake_soundfile_module = mock.Mock()
        fake_soundfile_module.SoundFile.return_value = BrokenSoundFile()
        session._recorder = Recorder()
        session._reader_thread = threading.Thread(target=session._reader_loop)
        session._writer_thread = threading.Thread(target=session._writer_loop)

        with mock.patch.object(recorder_module, "sf", fake_soundfile_module):
            session._reader_thread.start()
            self.assertTrue(reader_started.wait(1.0))
            session._writer_thread.start()
            self.assertTrue(writer_started.wait(1.0))
            release_writer.set()
            session._writer_thread.join(1.0)
            session._reader_thread.join(1.0)

            self.assertFalse(session._writer_thread.is_alive())
            self.assertFalse(session._reader_thread.is_alive())
            self.assertLessEqual(session._queue.qsize(), recorder_module._QUEUE_MAX_BLOCKS)
            with self.assertRaises(recorder_module.NativeRecorderError) as caught:
                session.stop()

        self.assertIs(caught.exception.__cause__, original)

    def test_stuck_reader_is_reported_and_reference_is_retained(self):
        session = self._session(channels=1)
        stuck = _StuckWorker()
        session._reader_thread = stuck

        with mock.patch.object(recorder_module, "_THREAD_STOP_TIMEOUT_SECONDS", 0.0), mock.patch.object(
            recorder_module, "_READER_STOP_GRACE_SECONDS", 0.0
        ):
            with self.assertRaisesRegex(recorder_module.NativeRecorderError, "toujours actif"):
                session.stop()

        self.assertIs(session._reader_thread, stuck)
        self.assertGreaterEqual(stuck.join_calls, 1)

    def test_corrupt_output_file_is_rejected(self):
        session = self._session(channels=1)
        session.output_path.write_bytes(b"not-a-wave-file")
        session._frames_written = 1

        with self.assertRaisesRegex(recorder_module.NativeRecorderError, "illisible"):
            session.stop()


class NativeRecorderCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)

    def test_pick_device_caps_multichannel_loopback_to_stereo(self):
        class Speaker:
            id = "speaker-id"
            name = "speaker"
            channels = 8

        class Microphone:
            id = "microphone-id"
            channels = 8

        fake_sc = mock.Mock()
        fake_sc.default_speaker.return_value = Speaker()
        fake_sc.get_microphone.return_value = Microphone()

        recorder = recorder_module.NativeRecorder(self.base_dir)
        with mock.patch.object(recorder_module, "sc", fake_sc):
            device_id, samplerate, channels = recorder._pick_device()

        self.assertEqual(device_id, "microphone-id")
        self.assertEqual(samplerate, 48_000.0)
        self.assertEqual(channels, 2)

    def test_init_purges_old_orphan_but_preserves_recent_and_unrelated_files(self):
        old_orphan = self.base_dir / f"native_recording_{uuid.uuid4().hex}.wav"
        recent_orphan = self.base_dir / f"native_recording_{uuid.uuid4().hex}.wav"
        unrelated = self.base_dir / "native_recording_notes.wav"
        for path in (old_orphan, recent_orphan, unrelated):
            path.write_bytes(b"data")
        old_timestamp = time.time() - 7_200
        os.utime(old_orphan, (old_timestamp, old_timestamp))
        os.utime(unrelated, (old_timestamp, old_timestamp))

        recorder_module.NativeRecorder(self.base_dir)

        self.assertFalse(old_orphan.exists())
        self.assertTrue(recent_orphan.exists())
        self.assertTrue(unrelated.exists())

    def test_failed_stop_clears_inactive_session_but_keeps_live_one_busy(self):
        class FailedSession:
            recording_id = "id"
            output_path = Path("unused.wav")

            def __init__(self, live):
                self.live = live
                self.discarded = False

            def stop(self):
                raise recorder_module.NativeRecorderError("boom")

            def has_live_workers(self):
                return self.live

            def discard_output(self):
                self.discarded = True

        recorder = recorder_module.NativeRecorder(self.base_dir)
        dead_session = FailedSession(live=False)
        recorder._active = dead_session
        with self.assertRaises(recorder_module.NativeRecorderError):
            recorder.stop()
        self.assertIsNone(recorder._active)
        self.assertTrue(dead_session.discarded)

        live_session = FailedSession(live=True)
        recorder._active = live_session
        with self.assertRaises(recorder_module.NativeRecorderError):
            recorder.stop()
        self.assertIs(recorder._active, live_session)
        self.assertFalse(live_session.discarded)


if __name__ == "__main__":
    unittest.main()
