"""Serializable transcription options; credentials are deliberately separate."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.config import LANGS, MODELS_CLOUD, MODELS_LOCAL, OUTPUT_PROMPTS
from .jobs import ValidationError


@dataclass(frozen=True)
class TranscriptionOptions:
    mode: str = "local"
    model: str = "base"
    language: str | None = "fr"
    output_type: str = "transcription"
    output_name: str = ""

    @property
    def use_api(self) -> bool:
        return self.mode in {"api", "openai"}

    def validate(self, api_key: str = "") -> None:
        if self.mode not in {"local", "api", "openai"}:
            raise ValidationError("Mode de transcription inconnu.")
        choices = MODELS_CLOUD if self.use_api else MODELS_LOCAL.values()
        if self.model not in choices:
            raise ValidationError("Modèle de transcription inconnu.")
        if self.language is not None and self.language not in LANGS.values():
            raise ValidationError("Langue inconnue.")
        if self.output_type not in {"transcription", *OUTPUT_PROMPTS}:
            raise ValidationError("Format de sortie inconnu.")
        if (self.use_api or self.output_type != "transcription") and not api_key.strip():
            raise ValidationError("Une clé OpenAI est nécessaire pour ce traitement.")

