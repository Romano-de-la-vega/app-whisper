"""OpenAI client construction and ordered contextual transcription."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from ..core.config import OPENAI_MAX_RETRIES, OPENAI_TIMEOUT_SECONDS
from ..models.jobs import ValidationError


def make_openai_client(api_key: str):
    if not api_key.strip():
        raise ValidationError("Aucune clé OpenAI fournie.")
    from openai import OpenAI

    return OpenAI(api_key=api_key.strip(), timeout=OPENAI_TIMEOUT_SECONDS,
                  max_retries=OPENAI_MAX_RETRIES)


def transcribe_chunks(client, chunks: Iterable[Path], model: str, language: str | None,
                      cancel_check: Callable[[], None], on_chunk: Callable[[int, str], None]) -> str:
    """Persist each completed response before honouring cancellation of the next."""
    texts: list[str] = []
    for index, path in enumerate(chunks, start=1):
        cancel_check()
        arguments = {"model": model}
        if language:
            arguments["language"] = language
        if texts:
            arguments["prompt"] = texts[-1][-500:]
        with path.open("rb") as stream:
            response = client.audio.transcriptions.create(file=stream, **arguments)
        text = (response if isinstance(response, str) else getattr(response, "text", "") or "").strip()
        if text:
            texts.append(text)
        on_chunk(index, "\n\n".join(texts).strip())
        cancel_check()
    return "\n\n".join(texts).strip()

