"""The existing CPU/int8 faster-whisper backend, loaded on demand."""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def load_model(model_path: Path):
    from faster_whisper import WhisperModel

    return WhisperModel(str(model_path), device="cpu", compute_type="int8",
                        local_files_only=True)


def transcribe(model, path: Path, language: str | None,
               cancel_check: Callable[[], None], on_segment: Callable[[float, str], None]) -> str:
    segments, info = model.transcribe(str(path), language=language, beam_size=5, vad_filter=True)
    duration = info.duration or 1.0
    lines: list[str] = []
    for segment in segments:
        cancel_check()
        text = (segment.text or "").strip()
        if text:
            lines.append(text)
        on_segment(min(max(0.0, segment.end) / duration, 1.0), "\n".join(lines).strip())
    cancel_check()
    return "\n".join(lines).strip()
