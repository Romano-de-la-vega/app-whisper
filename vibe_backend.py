from typing import Any


def load_model(lang: str) -> Any:
    """
    Charger et retourner un modèle Vibe prêt à transcrire.

    Implémentation à adapter selon https://github.com/thewh1teagle/vibe
    Exemples possibles (à titre indicatif):
      - Installation: pip install git+https://github.com/thewh1teagle/vibe.git
      - Chargement:   model = vibe.load_model(lang=lang)
    """
    raise RuntimeError(
        "Backend Vibe non configuré. Installez le paquet Vibe et implémentez load_model()/transcribe() dans vibe_backend.py"
    )


def transcribe(model: Any, path: str, lang: str) -> str:
    """
    Transcrire un fichier audio `path` en texte.
    Retourne la transcription en chaîne.
    """
    raise RuntimeError(
        "Backend Vibe non configuré. Implémentez transcribe(model, path, lang) dans vibe_backend.py"
    )

