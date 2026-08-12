from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import Future
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import numpy as np

import server


def create_audio_mp4(path: Path, duration_seconds: float = 2.2) -> None:
    sample_rate = 48_000
    frame_samples = 1024
    frame_count = int(duration_seconds * sample_rate / frame_samples) + 1
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        pts = 0
        for _ in range(frame_count):
            samples = np.zeros((1, frame_samples), dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = pts
            frame.time_base = Fraction(1, sample_rate)
            pts += frame_samples
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def create_video_without_audio(path: Path) -> None:
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


class MediaPipelineTests(unittest.TestCase):
    def test_api_chunk_limit_is_ten_minutes_with_encoding_margin(self) -> None:
        self.assertEqual(server.MAX_API_CHUNK_SECONDS, 600)
        self.assertGreater(server.API_CHUNK_CONTENT_SECONDS, 0)
        self.assertLess(server.API_CHUNK_CONTENT_SECONDS, server.MAX_API_CHUNK_SECONDS)

    def test_prepare_api_chunks_rejects_mp3_with_unknown_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp3"
            source.write_bytes(b"source")
            temp_dir = root / "tmp"

            def fake_split(_source, chunk_dir, _output_stem, cancel_check=None):
                chunk = chunk_dir / "source_part001.mp3"
                chunk.write_bytes(b"mp3")
                return [chunk]

            def fake_probe(path):
                return 42.0 if Path(path) == source else None

            with (
                patch.object(server, "TEMP_DIR", temp_dir),
                patch.object(server, "_split_media_to_mp3", side_effect=fake_split),
                patch.object(server, "_probe_audio_duration", side_effect=fake_probe),
            ):
                with self.assertRaisesRegex(RuntimeError, "(?i)durée"):
                    server._prepare_api_chunks("job-id", 1, source, "source")

            self.assertFalse((temp_dir / "job-id" / "api_chunks_001").exists())

    def test_mp4_is_converted_to_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reunion.mp4"
            destination = root / "reunion.mp3"
            create_audio_mp4(source)

            result = server._convert_media_to_mp3(source, destination)

            self.assertEqual(result, destination)
            self.assertGreater(destination.stat().st_size, 0)
            self.assertEqual(destination.suffix, ".mp3")
            duration = server._probe_audio_duration(destination)
            self.assertIsNotNone(duration)
            self.assertGreater(duration, 2.0)

    def test_mp3_parts_are_bounded_and_keep_all_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "longue.mp4"
            create_audio_mp4(source, duration_seconds=2.4)

            parts = server._transcode_to_mp3_parts(
                source,
                lambda index: root / f"part_{index:03d}.mp3",
                max_content_seconds=1,
            )

            self.assertGreaterEqual(len(parts), 3)
            durations = [server._probe_audio_duration(part) or 0 for part in parts]
            self.assertTrue(all(part.suffix == ".mp3" for part in parts))
            self.assertTrue(all(duration <= 1.2 for duration in durations), durations)
            self.assertGreater(sum(durations), 2.4)

    def test_video_without_audio_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "muette.mp4"
            create_video_without_audio(source)

            with self.assertRaisesRegex(RuntimeError, "aucune piste audio"):
                server._convert_media_to_mp3(source, root / "muette.mp3")

    def test_upload_name_is_sanitized_and_validated(self) -> None:
        safe = server._safe_upload_name(r"..\..\réunion?.MP4", 1)
        self.assertEqual(safe, "001_réunion_.mp4")
        with self.assertRaises(server.HTTPException) as raised:
            server._safe_upload_name("notes.exe", 2)
        self.assertEqual(raised.exception.status_code, 415)

    def test_huggingface_download_uses_current_supported_arguments(self) -> None:
        calls = []
        job_id = "c" * 32

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)

        with server.JOBS_LOCK:
            server.JOBS[job_id] = {"cancel_requested": False, "logs": []}
        try:
            with patch.object(server, "snapshot_download", fake_snapshot_download):
                server._snapshot_try_candidates(job_id, "org/model", "target")
        finally:
            with server.JOBS_LOCK:
                server.JOBS.pop(job_id, None)

        self.assertEqual(calls[0]["repo_id"], "org/model")
        self.assertNotIn("resume_download", calls[0])
        self.assertNotIn("local_dir_use_symlinks", calls[0])

    def test_public_status_never_exposes_internal_paths(self) -> None:
        snapshot = server._public_job_snapshot(
            {
                "status": "done",
                "cancel_requested": False,
                "files": [
                    {
                        "name": "réunion.mp4",
                        "path": r"C:\secret\001_reunion.mp4",
                        "stored_name": "001_reunion.mp4",
                        "output_stem": "reunion",
                        "out_path": r"C:\secret\reunion_transcription.txt",
                    }
                ],
            }
        )
        self.assertNotIn("cancel_requested", snapshot)
        self.assertNotIn("path", snapshot["files"][0])
        self.assertNotIn("stored_name", snapshot["files"][0])
        self.assertEqual(snapshot["files"][0]["output_name"], "reunion_transcription.txt")

    def test_pending_job_can_be_cancelled_without_deadlock(self) -> None:
        job_id = "b" * 32
        future = Future()
        job = {
            "status": "pending",
            "cancel_requested": False,
            "logs": [],
            "files": [{"status": "queued", "stage": "queued"}],
        }
        self.assertTrue(server.JOB_CAPACITY.acquire(blocking=False))
        with server.JOBS_LOCK:
            server.JOBS[job_id] = job
            server.JOB_FUTURES[job_id] = future
        future.add_done_callback(
            lambda completed: server._forget_job_future(job_id, completed)
        )
        try:
            response = server.cancel_job(job_id)
            self.assertEqual(response["status"], "cancelled")
            self.assertTrue(future.cancelled())
            self.assertEqual(job["status"], "cancelled")
            self.assertEqual(job["files"][0]["status"], "cancelled")
        finally:
            with server.JOBS_LOCK:
                server.JOBS.pop(job_id, None)
                server.JOB_FUTURES.pop(job_id, None)

    def test_cloud_segments_are_sent_and_recomposed_in_order(self) -> None:
        class FakeTranscriptions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, *, model, file, language, prompt=None):
                name = Path(file.name).name
                self.calls.append((model, name, language, prompt))
                return SimpleNamespace(text=f"texte {name}")

        class FakeClient:
            def __init__(self) -> None:
                self.transcriptions = FakeTranscriptions()
                self.audio = SimpleNamespace(transcriptions=self.transcriptions)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            chunk_dir = root / "chunks"
            chunk_dir.mkdir()
            chunks = [chunk_dir / "part_001.mp3", chunk_dir / "part_002.mp3"]
            for chunk in chunks:
                chunk.write_bytes(b"mp3")

            job_id = "a" * 32
            job = {
                "status": "running",
                "cancel_requested": False,
                "use_api": True,
                "model": "gpt-4o-mini-transcribe",
                "lang": "fr",
                "output_type": "transcription",
                "progress": 0.0,
                "logs": [],
                "files": [
                    {
                        "name": "source.mp4",
                        "stored_name": "001_source.mp4",
                        "output_stem": "source",
                        "path": str(source),
                        "status": "queued",
                        "stage": "queued",
                        "segment_index": 0,
                        "segment_count": 0,
                        "progress": 0.0,
                        "out_path": None,
                        "error": None,
                    }
                ],
            }
            client = FakeClient()
            with server.JOBS_LOCK:
                server.JOBS[job_id] = job
            try:
                with (
                    patch.object(server, "TRANS_DIR", root / "outputs"),
                    patch.object(
                        server,
                        "_prepare_api_chunks",
                        return_value=(chunks, chunk_dir, 700.0),
                    ),
                ):
                    server._run_cloud(job_id, client)

                output = Path(job["files"][0]["out_path"])
                self.assertEqual(
                    output.read_text(encoding="utf-8"),
                    "texte part_001.mp3\n\ntexte part_002.mp3\n",
                )
                self.assertEqual(
                    [call[1] for call in client.transcriptions.calls],
                    ["part_001.mp3", "part_002.mp3"],
                )
                self.assertIsNone(client.transcriptions.calls[0][3])
                self.assertEqual(client.transcriptions.calls[1][3], "texte part_001.mp3")
                self.assertEqual(job["files"][0]["status"], "done")
                self.assertEqual(job["files"][0]["segment_index"], 2)
                self.assertEqual(job["files"][0]["segment_count"], 2)
            finally:
                with server.JOBS_LOCK:
                    server.JOBS.pop(job_id, None)

    def test_run_job_releases_capacity_when_marker_touch_fails(self) -> None:
        capacity = Mock()
        marker_error = OSError("marker inaccessible")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(server, "JOB_CAPACITY", capacity),
                patch.object(server, "JOB_MARKER_DIR", root / "markers"),
                patch.object(server, "UPLOAD_DIR", root / "uploads"),
                patch.object(server, "TEMP_DIR", root / "tmp"),
                patch.object(server, "run_job"),
                patch.object(server.Path, "touch", side_effect=marker_error),
            ):
                try:
                    server._run_job_from_queue("job-id", None)
                except OSError as raised:
                    self.assertIs(raised, marker_error)

        capacity.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
