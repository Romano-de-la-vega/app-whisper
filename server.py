import os
import sys
import shutil
import uuid
import pathlib
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ----- LOCAL (faster-whisper)
from faster_whisper import WhisperModel

# ----- CLOUD (OpenAI) — optionnel
try:
    from openai import OpenAI  # lib openai >= 1.x
except Exception:
    OpenAI = None  # type: ignore

# ----- Téléchargement Hugging Face (pour montrer la progression)
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None  # type: ignore


# ========= Base dir compatible PyInstaller =========
def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)           # one-file (temp)
        return Path(sys.executable).parent      # one-folder
    return Path(__file__).resolve().parent       # dev

BASE_DIR = get_base_dir()

# ========= Dossiers de travail =========
UPLOAD_DIR = BASE_DIR / "uploads"
TRANS_DIR  = BASE_DIR / "transcriptions"
TEMP_DIR   = BASE_DIR / "tmp"
ASSETS_DIR = BASE_DIR / "assets"

# Répertoire persistant pour les modèles Whisper.
# Peut être personnalisé via la variable d'environnement WHISPER_MODELS_DIR.
def _get_models_dir() -> Path:
    env = os.getenv("WHISPER_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    # Dossier par défaut dans ~/.cache pour éviter les redécoupages du one-file PyInstaller.
    return Path.home() / ".cache" / "transcripteur-whisper" / "models"

MODELS_DIR = _get_models_dir()

for d in (UPLOAD_DIR, TRANS_DIR, TEMP_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ========= App, statiques & templates =========
app = FastAPI(title="Transcripteur Whisper (Web)")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), check_dir=False), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ========= Logging + handler global =========
import logging
logging.basicConfig(
    filename=str(BASE_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

@app.exception_handler(Exception)
async def all_errors(request: Request, exc: Exception):
    import traceback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logging.error("Unhandled error on %s: %s\n%s", request.url.path, exc, tb)
    return PlainTextResponse("Internal Server Error\n\n" + tb, status_code=500)

# ========= Health =========
@app.get("/health")
def health():
    return {"ok": True, "base_dir": str(BASE_DIR)}

# ========= Paramètres =========
MODELS_LOCAL: Dict[str, str] = {
    "Base": "base",
    "Small": "small",
    "Medium": "medium",
    "Large v3 (CPU lourd)": "large-v3",
}
MODELS_CLOUD: List[str] = ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]

LANGS: Dict[str, str] = {
    "Français": "fr", "Anglais": "en", "Espagnol": "es", "Allemand": "de",
    "Italien": "it", "Portugais": "pt", "Néerlandais": "nl", "Russe": "ru",
    "Arabe": "ar", "Chinois": "zh", "Japonais": "ja",
}
DEFAULT_LANG = "Français"
DEFAULT_MODEL_LOCAL = "Large v3 (CPU lourd)"

# — prompts selon le format de sortie désiré
OUTPUT_PROMPTS: Dict[str, str] = {
    "resume": (
        "Résume le texte suivant en français, sans phrase d'introduction ni conclusion:\n{texte}"
    ),
    "compte_rendu": (
        "Rédige un compte-rendu en français du texte suivant, sans phrase d'introduction ni conclusion:\n{texte}"
    ),
    "cahier_des_charges": (
        "À partir du texte suivant, rédige un cahier des charges en français, sans phrase d'introduction ni conclusion:\n{texte}"
    ),
    "notes_de_cadrage": (
        "À partir du texte suivant, rédige des notes de cadrage en français, sans phrase d'introduction ni conclusion:\n{texte}"
    ),
}

# — tailles approximatives pour le suivi de progression (octets)
#   valeurs proches des poids CTranslate2 (pratique pour une jauge réaliste)
MODEL_APPROX_SIZE = {
    "base":     150 * 1024**2,   # ~150 MB
    "small":    470 * 1024**2,   # ~470 MB
    "medium":  1500 * 1024**2,   # ~1.5 GB
    "large-v2": 2900 * 1024**2,  # ~2.9 GB
    "large-v3": 3100 * 1024**2,  # ~3.1 GB
}

# — mapping nom → repo HF faster-whisper
REPO_MAP = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# ========= Patch VAD (local) =========
def _ensure_vad_assets():
    src_assets = ASSETS_DIR
    dst_assets = Path.home() / ".cache" / "faster-whisper" / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)
    for fname in ("silero_vad.onnx", "silero_encoder_v5.onnx", "silero_decoder_v5.onnx"):
        s = src_assets / fname
        d = dst_assets / fname
        try:
            if s.exists() and not d.exists():
                shutil.copy2(s, d)
        except Exception as e:
            print(f"[WARN] Copie VAD échouée ({fname}): {e}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Evite les barres tqdm de HF dans la console
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_ensure_vad_assets()

# ========= Page d’accueil =========
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models_local": list(MODELS_LOCAL.keys()),
            "models_cloud": MODELS_CLOUD,
            "langs": list(LANGS.keys()),
            "DEFAULT_MODEL_LOCAL": DEFAULT_MODEL_LOCAL,
            "DEFAULT_LANG": DEFAULT_LANG,
        },
    )

# ========= Jobs =========
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

@app.post("/api/transcribe")
async def transcribe_endpoint(
    use_api: str = Form("0"),
    api_key: Optional[str] = Form(None),
    model_label: str = Form(...),
    lang_label: str = Form(...),
    output_type: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier envoyé")

    use_api_bool = (use_api == "1")

    if use_api_bool:
        if model_label not in MODELS_CLOUD:
            raise HTTPException(status_code=400, detail="Modèle API inconnu")
        model_name = model_label
        if OpenAI is None:
            raise HTTPException(status_code=400, detail="Le package 'openai' n'est pas installé côté serveur.")
    else:
        if model_label not in MODELS_LOCAL:
            raise HTTPException(status_code=400, detail="Modèle local inconnu")
        model_name = MODELS_LOCAL[model_label]

    if lang_label not in LANGS:
        raise HTTPException(status_code=400, detail="Langue inconnue")
    lang_code = LANGS[lang_label]

    if use_api_bool:
        if not output_type:
            output_type = "resume"
        if output_type not in OUTPUT_PROMPTS:
            raise HTTPException(status_code=400, detail="Format de sortie inconnu")
    else:
        output_type = None

    job_id = str(uuid.uuid4())
    job_upload_dir = UPLOAD_DIR / job_id
    job_trans_dir = TRANS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_trans_dir.mkdir(parents=True, exist_ok=True)

    files_meta = []
    for f in files:
        dest = job_upload_dir / f.filename
        with dest.open("wb") as out:
            out.write(await f.read())
        files_meta.append({
            "name": f.filename,
            "path": str(dest),
            "status": "queued",
            "progress": 0.0,
            "out_path": None,
            "error": None,
        })

    job: Dict[str, Any] = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "use_api": use_api_bool,
        "model": model_name,
        "lang": lang_code,
        "output_type": output_type,
        "progress": 0.0,
        "logs": [f"Job {job_id} créé avec {len(files_meta)} fichier(s)."],
        "files": files_meta,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=run_job, args=(job_id, api_key), daemon=True).start()
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job introuvable")
        return JSONResponse(job)

@app.get("/api/download/{job_id}")
def download_zip(job_id: str):
    job_trans_dir = TRANS_DIR / job_id
    if not job_trans_dir.exists():
        raise HTTPException(status_code=404, detail="Transcriptions introuvables")
    zip_path = TEMP_DIR / f"transcriptions_{job_id}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=job_trans_dir)
    return FileResponse(
        path=str(zip_path),
        filename=zip_path.name,
        media_type="application/zip",
        headers={"Cache-Control": "no-store"},
    )

@app.get("/api/download-txt/{job_id}")
def download_txt(job_id: str, kind: str = "transcription", merge: bool = True):
    job_trans_dir = TRANS_DIR / job_id
    if not job_trans_dir.exists():
        raise HTTPException(status_code=404, detail="Transcriptions introuvables")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    output_type = job.get("output_type") if job else None

    if kind == "summary":
        if not output_type:
            raise HTTPException(status_code=404, detail="Aucun résumé disponible")
        txt_files = sorted(job_trans_dir.glob(f"*_{output_type}.txt"))
    else:
        if output_type:
            txt_files = sorted(
                p for p in job_trans_dir.glob("*.txt") if not p.name.endswith(f"_{output_type}.txt")
            )
        else:
            txt_files = sorted(job_trans_dir.glob("*.txt"))

    if not txt_files:
        raise HTTPException(status_code=404, detail="Aucun .txt trouvé")

    name_root = output_type if kind == "summary" and output_type else "transcriptions"
    out_txt = TEMP_DIR / f"{name_root}_{job_id}.txt"

    if merge or len(txt_files) > 1:
        with out_txt.open("w", encoding="utf-8", newline="\n") as out:
            for i, p in enumerate(txt_files, 1):
                out.write(f"===== {p.name} =====\n")
                content = p.read_text(encoding="utf-8")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                if i < len(txt_files):
                    out.write("\n")
    else:
        out_txt = txt_files[0]

    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": f'attachment; filename="{out_txt.name}"',
    }
    return FileResponse(
        path=str(out_txt),
        media_type="text/plain; charset=utf-8",
        headers=headers,
        filename=out_txt.name,
    )

# ========= Worker principal =========
def run_job(job_id: str, api_key: Optional[str]):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return

    try:
        set_job_status(job_id, "running")
        if job["use_api"]:
            append_log(job_id, f"Mode API OpenAI · modèle: {job['model']} · langue: {job['lang']}")
            client = _make_openai_client(api_key)
            _run_cloud(job_id, client)
        else:
            append_log(job_id, f"Mode local (CPU int8) · modèle: {job['model']} · langue: {job['lang']}")
            _run_local(job_id)

        set_job_progress(job_id, 1.0)
        set_job_status(job_id, "done")
        append_log(job_id, "Tous les fichiers ont été traités. Vous pouvez télécharger les résultats.")
    except Exception as e:
        set_job_status(job_id, "error")
        append_log(job_id, f"[ERREUR JOB] {e}")

# ========= OpenAI (cloud) =========
def _make_openai_client(api_key: Optional[str]):
    if OpenAI is None:
        raise RuntimeError("Le package 'openai' n'est pas installé côté serveur.")
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Aucune clé API fournie (champ vide et OPENAI_API_KEY non défini).")
    return OpenAI(api_key=key)

def _run_cloud(job_id: str, client: "OpenAI"):
    with JOBS_LOCK:
        job = JOBS[job_id]
    total = len(job["files"])
    model_name = job["model"]
    lang = job["lang"]
    output_type = job.get("output_type", "resume")

    for idx, fmeta in enumerate(job["files"]):
        update_file_status(job_id, idx, "running")
        append_log(job_id, f"→ Envoi à OpenAI : {fmeta['name']}")
        try:
            with open(fmeta["path"], "rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=model_name,
                    file=fh,
                    language=lang,
                )
            text = (getattr(resp, "text", "") or "").strip()

            processed = text
            prompt_tmpl = OUTPUT_PROMPTS.get(output_type)
            if prompt_tmpl and text:
                append_log(job_id, f"→ GPT-4 pour '{output_type}'")
                prompt = prompt_tmpl.format(texte=text)
                resp2 = client.responses.create(model="gpt-4o", input=prompt)
                processed = (getattr(resp2, "output_text", "") or "").strip()

            out_dir = TRANS_DIR / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = pathlib.Path(fmeta["name"]).stem
            trans_file = out_dir / f"{stem}_transcription.txt"
            trans_file.write_text(
                text + ("\n" if text and not text.endswith("\n") else ""),
                encoding="utf-8",
            )

            out_file = trans_file
            if output_type and processed:
                out_file = out_dir / f"{stem}_{output_type}.txt"
                out_file.write_text(
                    processed + ("\n" if processed and not processed.endswith("\n") else ""),
                    encoding="utf-8",
                )

            set_file_output(job_id, idx, str(out_file))
            set_file_progress(job_id, idx, 1.0)                    # <— ajout
            set_job_progress(job_id, (idx + 1) / max(total, 1))    # <— ajout
            update_file_status(job_id, idx, "done")
            append_log(job_id, f"✓ Terminé (API) : {fmeta['name']} → {out_file.name}")

        except Exception as e:
            update_file_status(job_id, idx, "error", error=str(e))
            append_log(job_id, f"[ERREUR API] {fmeta['name']} : {e}")

        set_job_progress(job_id, (idx + 1) / max(total, 1))

# ========= Local (avec suivi de téléchargement) =========
def _run_local(job_id: str):
    with JOBS_LOCK:
        job = JOBS[job_id]

    # 1) S'assurer que le modèle est présent (sinon, on le télécharge avec suivi)
    model_name = job["model"]               # ex: "base", "large-v3", ...
    _ensure_local_model_with_progress(job_id, model_name)

    # 2) Transcription
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(MODELS_DIR))
    total = len(job["files"])
    append_log(job_id, "VAD: Silero · beam_size=5")

    for idx, fmeta in enumerate(job["files"]):
        update_file_status(job_id, idx, "running")
        append_log(job_id, f"→ Transcription locale : {fmeta['name']}")
        try:
            segments, info = model.transcribe(
                fmeta["path"],
                language=job["lang"],
                beam_size=5,
                vad_filter=True,
            )
            duration = info.duration or 1.0
            done = 0.0
            full_text: List[str] = []

            for seg in segments:
                done += max(0.0, (seg.end - seg.start))
                pct_file = min(done / duration, 1.0)
                set_file_progress(job_id, idx, pct_file)

                text = (seg.text or "").strip()
                if text:
                    append_log(job_id, text)
                    full_text.append(text)

                set_job_progress(job_id, (idx + pct_file) / max(total, 1))

            out_dir = TRANS_DIR / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_text = "\n".join(full_text).strip()
            out_file = out_dir / (pathlib.Path(fmeta["name"]).stem + "_transcription.txt")
            out_file.write_text(out_text + ("\n" if out_text and not out_text.endswith("\n") else ""), encoding="utf-8")

            set_file_output(job_id, idx, str(out_file))
            set_file_progress(job_id, idx, 1.0)                    # <— ajout
            set_job_progress(job_id, (idx + 1) / max(total, 1))    # <— ajout
            update_file_status(job_id, idx, "done")
            append_log(job_id, f"✓ Terminé (local) : {fmeta['name']} → {out_file.name}")

        except Exception as e:
            update_file_status(job_id, idx, "error", error=str(e))
            append_log(job_id, f"[ERREUR LOCAL] {fmeta['name']} : {e}")

# --- helpers de téléchargement + progression
def _ensure_local_model_with_progress(job_id: str, model_name: str):
    """Télécharge le modèle dans ./models si absent, en affichant la progression dans les logs."""
    # Si l’utilisateur a déjà le dossier du modèle, on ne fait rien
    target_dir = MODELS_DIR / model_name
    if target_dir.exists() and any(target_dir.rglob("*")):
        append_log(job_id, f"Modèle '{model_name}' déjà présent.")
        return

    append_log(job_id, f"Vérification du modèle '{model_name}'…")

    # Si huggingface_hub n'est pas dispo, on laisse faster-whisper gérer (pas de jauge)
    if snapshot_download is None:
        append_log(job_id, "[WARN] huggingface_hub indisponible → téléchargement sans jauge.")
        # Construction implicite fera le download_root (mais la progression ne sera pas affichée)
        WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(MODELS_DIR))
        return

    # Repo à télécharger
    repo_id = REPO_MAP.get(model_name)
    if not repo_id:
        # fallback: laisser FW gérer
        append_log(job_id, f"[WARN] Repo HF inconnu pour '{model_name}', téléchargement délégué.")
        WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(MODELS_DIR))
        return

    approx_total = MODEL_APPROX_SIZE.get(model_name, 1024**3)  # défaut 1 GB si inconnu
    append_log(job_id, f"Téléchargement du modèle (~{_fmt_size(approx_total)})…")

    # Thread “monitor” : calcule la taille du dossier pendant le download pour estimer %
    stop = threading.Event()
    def monitor():
        last_pct = -1
        while not stop.is_set():
            size = _dir_size_bytes(target_dir)
            pct = max(0, min(100, int((size / max(1, approx_total)) * 100)))
            if pct != last_pct:
                append_log(job_id, f"Téléchargement modèle : {pct}% ({_fmt_size(size)} / {_fmt_size(approx_total)})")
                last_pct = pct
            time.sleep(0.5)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    try:
        # Téléchargement bloquant dans ./models/<model_name>
        # (on force le chemin pour qu'il corresponde au nom du modèle)
        local_dir = MODELS_DIR / model_name
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns="*",
        )
        # petit flush final
        size = _dir_size_bytes(target_dir)
        append_log(job_id, f"Téléchargement modèle : 100% ({_fmt_size(size)})")
    finally:
        stop.set()
        mon.join(timeout=1.0)

def _dir_size_bytes(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except Exception:
        return 0

def _fmt_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"

# ========= Helpers thread-safe =========
def with_job(job_id: str, fn):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        fn(job)

def set_job_status(job_id: str, status: str):
    with_job(job_id, lambda j: j.__setitem__("status", status))

def set_job_progress(job_id: str, p: float):
    with_job(job_id, lambda j: j.__setitem__("progress", max(0.0, min(1.0, p))))

def append_log(job_id: str, message: str):
    with_job(job_id, lambda j: j["logs"].append(message))

def update_file_status(job_id: str, index: int, status: str, error: Optional[str] = None):
    def _upd(j):
        j["files"][index]["status"] = status
        if error:
            j["files"][index]["error"] = error
    with_job(job_id, _upd)

def set_file_progress(job_id: str, index: int, p: float):
    with_job(job_id, lambda j: j["files"][index].__setitem__("progress", max(0.0, min(1.0, p))))

def set_file_output(job_id: str, index: int, path: str):
    with_job(job_id, lambda j: j["files"][index].__setitem__("out_path", path))

# ========= Entrée (dev) =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)