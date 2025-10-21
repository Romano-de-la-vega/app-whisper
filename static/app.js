// ====== Sélecteurs DOM ======
const form = document.getElementById("form");
const modeSelect = document.getElementById("mode");
const apiKeyWrap = document.getElementById("api-key-wrap");
const apiKeyInput = document.getElementById("api_key");
const outputTypeWrap = document.getElementById("output-type-wrap");
const outputTypeSelect = document.getElementById("output_type");

const modelSelect = document.getElementById("model");
const langSelect = document.getElementById("lang");
const filesInput = document.getElementById("files");
const startBtn = document.getElementById("start");
const resetBtn = document.getElementById("reset");
const recordBtn = document.getElementById("record-btn");

const statusSection = document.getElementById("status");
const progressBar = document.getElementById("progress");
const jobIdSpan = document.getElementById("job-id");
const jobStateSpan = document.getElementById("job-state");
const filesList = document.getElementById("files-list");
const logsPre = document.getElementById("logs");
const downloadWrap = document.getElementById("downloads");
const summaryBtn = document.getElementById("btn-summary");
const estimateNode = document.getElementById("estimate");

const OUTPUT_LABELS = {
  transcription: "Télécharger la transcription (TXT)",
  resume: "Télécharger le résumé (TXT)",
  compte_rendu: "Télécharger le compte rendu (TXT)",
  note_de_cadrage: "Télécharger la note de cadrage (TXT)",
  cahier_des_charges: "Télécharger le cahier des charges (TXT)",
  procedure_technique: "Télécharger la procédure (TXT)",
  rapport_analyse: "Télécharger le rapport d'analyse (TXT)",
  support_formation: "Télécharger le support de formation (TXT)",
};

const themeBtn = document.getElementById("toggle-theme");
const logoImg = document.getElementById("logo");
const bodyEl = document.body;

// Masquer les boutons de téléchargement tant que la transcription n'est pas terminée
downloadWrap.hidden = true;

function updateSummaryBtnLabel() {
  const show = modeSelect.value === "api" && outputTypeSelect.value !== "transcription";
  summaryBtn.style.display = show ? "inline-flex" : "none";
  if (show) {
    summaryBtn.textContent = OUTPUT_LABELS[outputTypeSelect.value] || "Télécharger le document (TXT)";
  }
}
outputTypeSelect.addEventListener("change", updateSummaryBtnLabel);
updateSummaryBtnLabel();

// ====== État local ======
let pollTimer = null;
let currentJobId = null;
let lastLogLength = 0;
let isRunning = false;
let totalDurationMin = 0;

let particlesPromise = null;
function loadParticles() {
  if (!particlesPromise) {
    particlesPromise = import("./particles.js").then(() => window.Particles);
  }
  return particlesPromise;
}

const AudioContextClass = window.AudioContext || window.webkitAudioContext;
const RECORD_MIME_TYPE = (() => {
  if (typeof MediaRecorder === "undefined") return "";
  const preferred = "audio/webm;codecs=opus";
  if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(preferred)) return preferred;
  return "";
})();

let mediaRecorder = null;
let recordingChunks = [];
let recordingStreams = [];
let recordingContext = null;
let recordedFiles = [];
let recordButtonResetTimer = null;

function setRecordButtonState(state) {
  if (!recordBtn) return;
  if (state !== "idle" && recordButtonResetTimer) {
    clearTimeout(recordButtonResetTimer);
    recordButtonResetTimer = null;
  }
  recordBtn.dataset.state = state;
  if (state === "recording") {
    recordBtn.textContent = "⏹️ Arrêter";
  } else if (state === "preparing") {
    recordBtn.textContent = "⏳ Préparation…";
  } else {
    recordBtn.textContent = "⏺️ Enregistrer";
  }
}

function cleanupRecording(stopStreams = false) {
  if (stopStreams) {
    recordingStreams.forEach(stream => {
      stream.getTracks().forEach(track => track.stop());
    });
  }
  recordingStreams = [];
  if (recordingContext) {
    recordingContext.close().catch(() => {});
    recordingContext = null;
  }
  mediaRecorder = null;
  recordingChunks = [];
}

function allAudioFiles() {
  const map = new Map();
  const push = (file) => {
    if (!file) return;
    const key = `${file.name}::${file.size}::${file.lastModified}`;
    map.set(key, file);
  };
  Array.from(filesInput.files || []).forEach(push);
  recordedFiles.forEach(push);
  return Array.from(map.values());
}

function syncRecordedFilesToInput() {
  if (typeof DataTransfer === "undefined") return false;
  try {
    const transfer = new DataTransfer();
    allAudioFiles().forEach(file => transfer.items.add(file));
    filesInput.files = transfer.files;
    return true;
  } catch (err) {
    return false;
  }
}

async function startRecordingCapture() {
  if (!recordBtn) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !navigator.mediaDevices.getDisplayMedia) {
    alert("L'enregistrement système + micro n'est pas supporté par ce navigateur.");
    return;
  }
  if (!AudioContextClass) {
    alert("Votre navigateur ne permet pas de mixer les pistes audio nécessaires.");
    return;
  }

  recordBtn.disabled = true;
  setRecordButtonState("preparing");

  let micStream = null;
  let systemStream = null;

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 2,
        echoCancellation: false,
        noiseSuppression: false,
      },
      video: false,
    });

    systemStream = await navigator.mediaDevices.getDisplayMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
      },
      video: { frameRate: 5 },
    });

    systemStream.getVideoTracks().forEach(track => {
      track.enabled = false;
    });

    const sysAudioTracks = systemStream.getAudioTracks();
    if (!sysAudioTracks.length) {
      throw new Error("Aucun son système n'a été partagé.");
    }

    const audioCtx = new AudioContextClass();
    const destination = audioCtx.createMediaStreamDestination();

    const micSource = audioCtx.createMediaStreamSource(micStream);
    micSource.connect(destination);

    const systemAudioStream = new MediaStream(sysAudioTracks);
    const sysSource = audioCtx.createMediaStreamSource(systemAudioStream);
    sysSource.connect(destination);

    recordingStreams = [micStream, systemStream, systemAudioStream];
    recordingContext = audioCtx;
    recordingChunks = [];

    const options = RECORD_MIME_TYPE ? { mimeType: RECORD_MIME_TYPE } : undefined;
    mediaRecorder = new MediaRecorder(destination.stream, options);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size) recordingChunks.push(event.data);
    };
    mediaRecorder.onstop = () => {
      const mime = RECORD_MIME_TYPE || (mediaRecorder ? mediaRecorder.mimeType : "audio/webm");
      const blob = new Blob(recordingChunks, { type: mime || "audio/webm" });
      cleanupRecording(true);

      if (!blob.size) {
        setRecordButtonState("idle");
        recordBtn.disabled = false;
        alert("Aucun audio n'a été capturé.");
        return;
      }

      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const fileName = `enregistrement_${stamp}.webm`;
      const file = new File([blob], fileName, { type: blob.type, lastModified: Date.now() });
      recordedFiles.push(file);

      const synced = syncRecordedFilesToInput();
      if (synced) {
        recordBtn.title = "Enregistrer";
      } else {
        recordBtn.title = "Le fichier a été ajouté pour l'envoi, mais il peut ne pas apparaître dans la liste.";
        alert("L'enregistrement est prêt et sera envoyé automatiquement, même s'il n'apparaît pas dans la liste des fichiers.");
      }

      const changeEvent = new Event("change", { bubbles: true });
      filesInput.dispatchEvent(changeEvent);

      setRecordButtonState("idle");
      recordBtn.textContent = "✅ Ajouté";
      recordBtn.disabled = false;
      if (recordButtonResetTimer) clearTimeout(recordButtonResetTimer);
      recordButtonResetTimer = setTimeout(() => {
        recordButtonResetTimer = null;
        setRecordButtonState("idle");
      }, 2000);
    };

    mediaRecorder.start();
    setRecordButtonState("recording");
  } catch (err) {
    if (micStream) micStream.getTracks().forEach(track => track.stop());
    if (systemStream) systemStream.getTracks().forEach(track => track.stop());
    cleanupRecording();
    setRecordButtonState("idle");
    alert("Impossible de démarrer l'enregistrement : " + (err && err.message ? err.message : err));
  } finally {
    recordBtn.disabled = false;
  }
}

function stopRecordingCapture() {
  if (!mediaRecorder) return;
  if (mediaRecorder.state !== "inactive") {
    setRecordButtonState("preparing");
    recordBtn.disabled = true;
    mediaRecorder.stop();
  }
}

function setTranscribing(active) {
  bodyEl.classList.toggle('transcribing', !!active);
  loadParticles().then(p => {
    if (!p) return;
    if (active) p.start();
    else p.stop();
  });
}

// ====== Config serveur ======
(function initConfig() {
  const node = document.getElementById("whisper-config");
  const cfg = JSON.parse(node.textContent || "{}");
  window.MODELS_LOCAL = cfg.MODELS_LOCAL || [];
  window.MODELS_CLOUD = cfg.MODELS_CLOUD || [];
  window.LANGS = cfg.LANGS || [];
  window.DEFAULT_MODEL_LOCAL = cfg.DEFAULT_MODEL_LOCAL || (window.MODELS_LOCAL[0] || "");
  window.DEFAULT_LANG = cfg.DEFAULT_LANG || (window.LANGS[0] || "fr");
})();

// ====== Thème (persistance localStorage) ======
(function initTheme() {
  const root = document.documentElement;
  let current = root.getAttribute("data-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  root.setAttribute("data-theme", current);
  themeBtn.textContent = current === "dark" ? "☀️ Mode clair" : "🌙 Mode sombre";
  if (logoImg) logoImg.src = current === "dark" ? "/static/logo_white.png" : "/static/logo.png";

  themeBtn.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    themeBtn.textContent = next === "dark" ? "☀️ Mode clair" : "🌙 Mode sombre";
    if (logoImg) logoImg.src = next === "dark" ? "/static/logo_white.png" : "/static/logo.png";
  });
})();

// ====== Options ======
function fillModelOptions() {
  const useAPI = modeSelect.value === "api";
  const list = useAPI ? window.MODELS_CLOUD : window.MODELS_LOCAL;
  const def = useAPI ? (list[0] || "") : window.DEFAULT_MODEL_LOCAL;

  modelSelect.innerHTML = "";
  list.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    if (m === def) opt.selected = true;
    modelSelect.appendChild(opt);
  });

  apiKeyWrap.style.display = useAPI ? "flex" : "none";
  outputTypeWrap.style.display = useAPI ? "flex" : "none";
}
function fillLangOptions() {
  langSelect.innerHTML = "";
  (window.LANGS || []).forEach(l => {
    const opt = document.createElement("option");
    opt.value = l; opt.textContent = l;
    if (l === window.DEFAULT_LANG) opt.selected = true;
    langSelect.appendChild(opt);
  });
}

// ====== Estimation ======
const LOCAL_RATES = {
  "Base": 0.33,
  "Small": 0.88,
  "Medium": 2.4,
  "Large v3 (CPU lourd)": 4.9,
  "Large v3 Turbo (recommandé)": 1.75
};
const API_RATES = {
  "whisper-1": 0.006,
  "gpt-4o-transcribe": 0.006,
  "gpt-4o-mini-transcribe": 0.003
};

async function computeTotalDuration() {
  const files = allAudioFiles();
  if (!files.length) {
    totalDurationMin = 0;
    updateEstimate();
    return;
  }
  const durations = await Promise.all(files.map(getAudioDuration));
  totalDurationMin = durations.reduce((a, b) => a + b, 0) / 60;
  updateEstimate();
}

function getAudioDuration(file) {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(audio.duration || 0);
    };
    audio.onerror = () => resolve(0);
    audio.src = url;
  });
}

function updateEstimate() {
  if (!estimateNode) return;
  let text = "";
  if (totalDurationMin > 0) {
    const mode = modeSelect.value;
    const model = modelSelect.value;
    if (mode === "local" && LOCAL_RATES[model]) {
      const est = (totalDurationMin * LOCAL_RATES[model]).toFixed(2);
      text = `Estimation temps : ${est} min`;
    } else if (mode === "api" && API_RATES[model]) {
      const est = (totalDurationMin * API_RATES[model]).toFixed(2);
      text = `Coût estimé : ${est} €`;
    }
  }
  estimateNode.textContent = text;
}

// ====== Rendu ======
function formatPct(p) { return Math.round((p || 0) * 100); }

function renderFiles(files) {
  filesList.innerHTML = "";
  (files || []).forEach((f) => {
    const pct = f.status === "done" ? 100 : Math.min(100, formatPct(f.progress || 0));
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div class="name">${f.name}</div>
      <div class="state">État : ${f.status}${f.error ? " — " + f.error : ""}</div>
      <div class="row-progress"><div style="width:${pct}%"></div></div>
      ${f.out_path ? `<div class="state">Sortie : ${f.out_path.split("/").pop()}</div>` : ""}
    `;
    filesList.appendChild(row);
  });
}


function autoscrollLogs() {
  logsPre.scrollTop = logsPre.scrollHeight;
}

// ====== Téléchargements ======
async function downloadZip(jobId) {
  try {
    const res = await fetch(`/api/download/${jobId}`, { method: 'GET', cache: 'no-store' });
    if (!res.ok) { alert(`Échec ZIP (${res.status}).`); return; }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `transcriptions_${jobId}.zip`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) { alert('Échec du téléchargement ZIP : ' + e); }
}
window.downloadZip = downloadZip;

async function downloadTxt(jobId, kind = 'transcription', merge = true) {
  try {
    const res = await fetch(`/api/download-txt/${jobId}?merge=${merge ? 1 : 0}&kind=${kind}`, { method: 'GET', cache: 'no-store' });
    if (!res.ok) { alert(`Échec TXT (${res.status}).`); return; }
    const disposition = res.headers.get('Content-Disposition');
    let filename = `transcriptions_${jobId}.txt`;
    if (disposition) {
      const match = /filename="?([^";]+)"?/i.exec(disposition);
      if (match && match[1]) filename = match[1];
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) { alert('Échec du téléchargement TXT : ' + e); }
}
window.downloadTxt = downloadTxt;


// ====== Polling ======
async function pollStatus() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/status/${currentJobId}`);
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();

    jobIdSpan.textContent = `Job : ${currentJobId}`;
    jobStateSpan.textContent = job.status;
    progressBar.style.width = `${formatPct(job.progress)}%`;
    renderFiles(job.files);

    // Assurer l'affichage correct des boutons selon l'état et le mode
    downloadWrap.hidden = job.status !== "done";
    const showSummary = job.use_api && job.output_type && job.output_type !== "transcription";
    summaryBtn.style.display = showSummary ? "inline-flex" : "none";
    if (showSummary) {
      summaryBtn.textContent = OUTPUT_LABELS[job.output_type] || "Télécharger le document (TXT)";
    }

    if (Array.isArray(job.logs)) {
      const slice = job.logs.slice(lastLogLength).join("\n");
      if (slice.trim().length) {
        logsPre.textContent += (logsPre.textContent ? "\n" : "") + slice;
        lastLogLength = job.logs.length;
        autoscrollLogs();
      }
    }

    if (job.status === "done" || job.status === "error") {
      progressBar.style.width = "100%";
      clearInterval(pollTimer); pollTimer = null;
      isRunning = false;
      setTranscribing(false);
      startBtn.disabled = false;
      startBtn.textContent = "Lancer la transcription";
      startBtn.classList.remove("danger");
    }

    

  } catch (err) {
    console.error(err);
    clearInterval(pollTimer); pollTimer = null;
    isRunning = false;
    setTranscribing(false);
    startBtn.disabled = false;
    startBtn.textContent = "Lancer la transcription";
    startBtn.classList.remove("danger");
  }
}

// ====== Submit / Start-Stop ======
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (mediaRecorder && mediaRecorder.state === "recording") {
    alert("Arrête l'enregistrement avant de lancer la transcription.");
    return;
  }

  // === STOP (UI) ===
  if (isRunning) {
    // on affiche immédiatement l'état "arrêt en cours…"
    jobStateSpan.textContent = "arrêt en cours…";
    startBtn.disabled = true;            // gèle le bouton pendant qu'on arrête le polling
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;

    // petit délai visuel pour que l'utilisateur voie l'état
    setTimeout(() => {
      isRunning = false;
      setTranscribing(false);
      startBtn.disabled = false;
      startBtn.textContent = "Lancer la transcription";
      startBtn.classList.remove("danger");
      jobStateSpan.textContent = "arrêté (UI)";
    }, 500);

    return;
  }

  // === START ===
  const selectedFiles = allAudioFiles();

  if (!selectedFiles.length) {
    alert("Ajoute au moins un fichier audio.");
    return;
  }

  // lock UI + reset affichages
  isRunning = true;
  setTranscribing(true);
  startBtn.textContent = "Arrêter la transcription";
  startBtn.classList.add("danger");
  startBtn.disabled = true;      // on le réactive dès que le job démarre
  downloadWrap.hidden = true;
  logsPre.textContent = "";
  filesList.innerHTML = "";
  statusSection.hidden = false;
  progressBar.style.width = "0%";
  jobStateSpan.textContent = "démarrage…";
  jobIdSpan.textContent = "";
  lastLogLength = 0;

  const fd = new FormData();
  const use_api = modeSelect.value === "api";
  updateSummaryBtnLabel();
  fd.append("use_api", use_api ? "1" : "0");
  fd.append("api_key", (apiKeyInput.value || "").trim());
  fd.append("model_label", modelSelect.value);
  fd.append("lang_label", langSelect.value);
  if (use_api) fd.append("output_type", outputTypeSelect.value);
  selectedFiles.forEach(f => fd.append("files", f, f.name));


  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    currentJobId = data.job_id;
    jobIdSpan.textContent = `Job : ${currentJobId}`;
    jobStateSpan.textContent = "en cours";
    startBtn.disabled = false;   // on autorise l'arrêt (UI) maintenant que le job existe
    pollTimer = setInterval(pollStatus, 1000);
  } catch (err) {
    console.error(err);
    alert("Erreur au lancement : " + err.message);
    isRunning = false;
    setTranscribing(false);
    startBtn.disabled = false;
    startBtn.textContent = "Lancer la transcription";
    startBtn.classList.remove("danger");
  }
});

// ====== Réinitialiser (UI only) ======
resetBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecordingCapture();
  } else {
    cleanupRecording(true);
    setRecordButtonState("idle");
  }

  recordedFiles = [];
  syncRecordedFilesToInput();
  recordBtn && (recordBtn.title = "Enregistrer");
  if (recordButtonResetTimer) {
    clearTimeout(recordButtonResetTimer);
    recordButtonResetTimer = null;
  }

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  currentJobId = null;
  lastLogLength = 0;
  isRunning = false;
  setTranscribing(false);

  // reset visuel du formulaire
  form.reset();
  statusSection.hidden = true;
  logsPre.textContent = "";
  filesList.innerHTML = "";
  progressBar.style.width = "0%";
  downloadWrap.hidden = true;
  summaryBtn.style.display = "none";


  // Remettre les options par défaut
  fillModelOptions();
  fillLangOptions();
  estimateNode.textContent = "";
  totalDurationMin = 0;

  // Remettre le bouton principal
  startBtn.disabled = false;
  startBtn.textContent = "Lancer la transcription";
  startBtn.classList.remove("danger");
});

// ====== Init ======
modeSelect.addEventListener("change", () => { fillModelOptions(); updateEstimate(); updateSummaryBtnLabel(); });
modelSelect.addEventListener("change", updateEstimate);
filesInput.addEventListener("change", () => {
  if (recordedFiles.length) {
    syncRecordedFilesToInput();
  }
  computeTotalDuration();
});

fillModelOptions();
fillLangOptions();
updateEstimate();

if (recordBtn) {
  setRecordButtonState("idle");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia || !AudioContextClass) {
    recordBtn.disabled = true;
    recordBtn.title = "Votre navigateur ne permet pas cet enregistrement.";
  } else {
    recordBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecordingCapture();
      } else {
        startRecordingCapture();
      }
    });
  }
}
