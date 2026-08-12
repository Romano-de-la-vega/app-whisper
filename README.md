# Transcripteur Whisper

Application locale FastAPI/pywebview pour transcrire des fichiers audio ou vidéo avec `faster-whisper` (CPU) ou l'API OpenAI.

## Fonctions principales

- lots de fichiers audio et vidéo (`mp3`, `wav`, `m4a`, `flac`, `aac`, `ogg`, `mpga`, `mp4`, `webm`, `mkv`, `mov`) ;
- conversion automatique des vidéos, notamment MP4, vers un MP3 mono 16 kHz ;
- mode API : conversion systématique en MP3 64 kb/s, découpage en segments de **10 minutes maximum**, envoi séquentiel et recomposition dans l'ordre ;
- conservation d'une transcription partielle si un segment ultérieur échoue ;
- génération hiérarchique des résumés et documents pour les transcriptions très longues ;
- transcription locale sans envoi de données ;
- annulation coopérative, suivi détaillé et téléchargements TXT/ZIP ;
- enregistrement du son système sous Windows lorsque `SoundCard`/WASAPI est disponible.

PyAV embarque les bibliothèques de décodage/encodage utilisées par l'application : aucun exécutable `ffmpeg` séparé n'est requis. Le démarrage vérifie la présence de l'encodeur `libmp3lame`.

## Installation et lancement

Python 3.11 ou plus récent est recommandé.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main_gui.py
```

Pour lancer uniquement le serveur web :

```powershell
python server.py
```

Puis ouvrir `http://127.0.0.1:8000`.

En mode API, renseigner la clé dans l'interface ou définir `OPENAI_API_KEY`. La clé n'est ni enregistrée dans le job ni écrite dans les logs.

Les routes locales qui modifient l'état exigent un jeton éphémère injecté dans l'interface et contrôlent l'hôte/l'origine. Cela empêche une page web tierce de lancer un enregistrement ou de consommer la clé API définie dans l'environnement.

Le modèle API proposé par défaut est `gpt-transcribe`, conformément à la documentation OpenAI actuelle ; les modèles `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` et `whisper-1` restent sélectionnables.

## Traitement des fichiers longs avec l'API

Chaque média est décodé côté serveur puis converti en segments MP3 mono 16 kHz à 64 kb/s. Une marge d'encodage limite le contenu à 599 secondes, puis la durée conteneur est contrôlée avant chaque envoi. À ce débit, un segment de dix minutes pèse environ 4,8 Mo ; une limite de 24 Mo est également vérifiée.

Les segments sont envoyés un à un. La fin de la transcription précédente sert de contexte au segment suivant, et les textes sont assemblés dans le même ordre. Un fichier TXT intermédiaire est mis à jour après chaque réponse réussie.

## Données et configuration

En développement, les données sont placées dans le dépôt. Dans l'exécutable PyInstaller, elles sont placées dans `%LOCALAPPDATA%\TranscripteurWhisper` afin de ne pas écrire dans les ressources embarquées.

Variables facultatives :

- `WHISPER_DATA_DIR` : dossier de données ;
- `WHISPER_MODELS_DIR` : cache des modèles ;
- `WHISPER_MAX_UPLOAD_MB` : taille maximale d'un fichier (2048 par défaut) ;
- `WHISPER_MAX_JOB_UPLOAD_MB` : taille cumulée d'un lot (4096 par défaut) ;
- `WHISPER_MAX_FILES` : nombre maximal de fichiers par lot (50) ;
- `WHISPER_API_JOB_WORKERS` : traitements API parallèles (2) ;
- `WHISPER_MAX_QUEUED_JOBS` : traitements en attente (20) ;
- `WHISPER_JOB_RETENTION_HOURS` : durée de disponibilité des résultats (24 h, minimum 1 h).
- `WHISPER_DOCUMENT_MODEL` : modèle Responses pour les documents (`gpt-5-mini` par défaut) ;
- `WHISPER_DOCUMENT_CHUNK_CHARS` : budget prudent de chaque bloc documentaire (80 000 caractères) ;
- `WHISPER_OPENAI_TIMEOUT_SECONDS` : délai maximal d'un appel OpenAI (180 s, minimum 30 s) ;
- `WHISPER_OPENAI_MAX_RETRIES` : nouvelles tentatives SDK par appel (1 par défaut).

Les médias source et conversions temporaires d'un nouveau job sont supprimés dès sa fin. Les sorties sont purgées après la durée de rétention lors d'une nouvelle soumission. Un marqueur persistant permet aussi de nettoyer les jobs interrompus par un crash.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m compileall -q server.py native_recorder.py main_gui.py
```
