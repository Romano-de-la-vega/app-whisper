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
const recordingHint = document.getElementById("recording-hint");
const recordTimer = document.getElementById("record-timer");

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

let mediaRecorder = null;
let recordingChunks = [];
let recordingStreams = [];
let recordingAudioContext = null;
let recordingAudioSources = [];
let lastRecordedFile = null;
let isRecording = false;
let recordTimerInterval = null;
let recordStartTime = 0;

let particlesPromise = null;
function loadParticles() {
  if (!particlesPromise) {
    particlesPromise = import("./particles.js").then(() => window.Particles);
  }
  return particlesPromise;
}

function setTranscribing(active) {
  bodyEl.classList.toggle('transcribing', !!active);
  loadParticles().then(p => {
    if (!p) return;
    if (active) p.start();
    else p.stop();
  });
}

function setRecordButtonState(active) {
  if (!recordBtn) return;
  recordBtn.classList.toggle("is-recording", !!active);
  recordBtn.innerHTML = `<span class="dot" aria-hidden="true"></span>${active ? "Arrêter" : "Enregistrer"}`;
}

function resetRecordTimerDisplay() {
  if (!recordTimer) return;
  recordTimer.textContent = "00:00:00";
  recordTimer.classList.remove("is-recording");
}

function updateRecordTimerDisplay() {
  if (!recordTimer) return;
  const elapsed = Math.max(0, Math.floor((Date.now() - recordStartTime) / 1000));
  const hours = String(Math.floor(elapsed / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  recordTimer.textContent = `${hours}:${minutes}:${seconds}`;
}

function startRecordTimer() {
  if (!recordTimer) return;
  recordStartTime = Date.now();
  recordTimer.classList.add("is-recording");
  updateRecordTimerDisplay();
  if (recordTimerInterval) clearInterval(recordTimerInterval);
  recordTimerInterval = setInterval(updateRecordTimerDisplay, 500);
}

function stopRecordTimer(resetDisplay = false) {
  if (recordTimerInterval) {
    clearInterval(recordTimerInterval);
    recordTimerInterval = null;
  }
  if (!recordTimer) return;
  if (!resetDisplay) {
    updateRecordTimerDisplay();
  }
  recordTimer.classList.remove("is-recording");
  if (resetDisplay) {
    recordTimer.textContent = "00:00:00";
  }
}

function updateRecordingHint(text = "", isError = false) {
  if (!recordingHint) return;
  recordingHint.textContent = text;
  recordingHint.style.color = isError ? "#e33c3c" : "var(--muted)";
}

function stopRecordingStreams() {
  recordingStreams.forEach((stream) => {
    if (!stream) return;
    stream.getTracks().forEach(track => {
      try { track.stop(); } catch (_) { /* noop */ }
    });
  });
  recordingStreams = [];

  recordingAudioSources.forEach((source) => {
    try { source.disconnect(); } catch (_) { /* noop */ }
  });
  recordingAudioSources = [];

  if (recordingAudioContext) {
    try { recordingAudioContext.close(); } catch (_) { /* noop */ }
    recordingAudioContext = null;
  }
}

function buildDirectSystemAudioConstraints() {
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getSupportedConstraints !== "function") {
    return null;
  }
  const supported = navigator.mediaDevices.getSupportedConstraints();
  if (!supported.systemAudio) {
    return null;
  }

  const audio = { systemAudio: "include" };
  if (supported.suppressLocalAudioPlayback) audio.suppressLocalAudioPlayback = false;
  if (supported.channelCount) audio.channelCount = 2;
  if (supported.echoCancellation) audio.echoCancellation = false;
  if (supported.noiseSuppression) audio.noiseSuppression = false;

  return { audio, video: false };
}

async function requestSystemAudioStream() {
  if (!navigator.mediaDevices) {
    throw new Error("Capture audio non supportée.");
  }

  const directConstraints = buildDirectSystemAudioConstraints();
  if (directConstraints) {
    try {
      const directStream = await navigator.mediaDevices.getUserMedia(directConstraints);
      if (directStream && directStream.getAudioTracks().length) {
        return directStream;
      }
      directStream.getTracks().forEach(track => {
        try { track.stop(); } catch (_) { /* noop */ }
      });
    } catch (err) {
      console.warn("Direct system audio capture failed", err);
    }
  }

  if (typeof navigator.mediaDevices.getDisplayMedia !== "function") {
    throw new Error("La capture du son du PC n'est pas supportée par ce navigateur.");
  }

  const supported = navigator.mediaDevices.getSupportedConstraints ? navigator.mediaDevices.getSupportedConstraints() : {};
  let audioConstraints;
  if (supported.systemAudio) {
    audioConstraints = { systemAudio: "include" };
    if (supported.suppressLocalAudioPlayback) audioConstraints.suppressLocalAudioPlayback = false;
  } else {
    audioConstraints = true;
  }

  const systemStream = await navigator.mediaDevices.getDisplayMedia({
    audio: audioConstraints,
    video: false,
  });

  if (!systemStream || !systemStream.getAudioTracks().length) {
    throw new Error("Aucune piste audio système détectée.");
  }

  return systemStream;
}

async function startRecording() {
  if (!recordBtn) return;
  if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
    updateRecordingHint("Enregistrement non supporté sur ce navigateur.", true);
    return;
  }

  recordBtn.disabled = true;
  updateRecordingHint("Initialisation… autorisez le micro et le son du système.");

  const cleanupStreams = [];

  try {
    const micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 2,
      }
    });

    cleanupStreams.push(micStream);
    recordingStreams = cleanupStreams.slice();

    const systemStream = await requestSystemAudioStream();

    cleanupStreams.push(systemStream);
    recordingStreams = cleanupStreams.slice();

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      throw new Error("Mixage audio non supporté sur ce navigateur.");
    }

    recordingAudioContext = new AudioContextCtor();
    if (recordingAudioContext.state === "suspended") {
      await recordingAudioContext.resume().catch(() => {});
    }

    const destination = recordingAudioContext.createMediaStreamDestination();
    recordingAudioSources = [];

    if (systemStream && systemStream.getAudioTracks().length) {
      const systemSource = recordingAudioContext.createMediaStreamSource(systemStream);
      systemSource.connect(destination);
      recordingAudioSources.push(systemSource);
    }

    if (micStream && micStream.getAudioTracks().length) {
      const micSource = recordingAudioContext.createMediaStreamSource(micStream);
      micSource.connect(destination);
      recordingAudioSources.push(micSource);
    }

    if (!recordingAudioSources.length) {
      throw new Error("Aucune piste audio détectée.");
    }

    const mixedStream = destination.stream;

    cleanupStreams.push(mixedStream);
    recordingStreams = cleanupStreams.filter(Boolean);
    recordingChunks = [];

    let recorderOptions = undefined;
    if (typeof MediaRecorder.isTypeSupported === "function") {
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        recorderOptions = { mimeType: "audio/webm;codecs=opus" };
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        recorderOptions = { mimeType: "audio/webm" };
      }
    }

    mediaRecorder = recorderOptions ? new MediaRecorder(mixedStream, recorderOptions) : new MediaRecorder(mixedStream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordingChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = finalizeRecording;
    mediaRecorder.onerror = (event) => {
      console.error(event.error || event);
      updateRecordingHint("Erreur d'enregistrement : " + (event.error ? event.error.message : event.message || event.type), true);
      isRecording = false;
      setRecordButtonState(false);
      if (recordBtn) recordBtn.disabled = false;
      stopRecordingStreams();
      mediaRecorder = null;
      recordingChunks = [];
      stopRecordTimer(true);
    };

    mediaRecorder.start();
    isRecording = true;
    setRecordButtonState(true);
    updateRecordingHint("Enregistrement en cours… micro et son du PC capturés.");
    startRecordTimer();
  } catch (err) {
    console.error(err);
    let message = "Impossible de démarrer l'enregistrement.";
    if (err && err.name === "NotAllowedError") {
      message = "Autorisation refusée. Veuillez autoriser le micro et le son du système.";
    } else if (err && err.message) {
      message = `${message} ${err.message}`;
    }
    updateRecordingHint(message, true);
    cleanupStreams.forEach((stream) => {
      if (!stream) return;
      stream.getTracks().forEach(track => {
        try { track.stop(); } catch (_) { /* noop */ }
      });
    });
    stopRecordingStreams();
    mediaRecorder = null;
    recordingChunks = [];
    isRecording = false;
    setRecordButtonState(false);
    stopRecordTimer(true);
  } finally {
    recordBtn.disabled = false;
  }
}

function stopRecordingAction() {
  if (!isRecording || !mediaRecorder) return;
  recordBtn.disabled = true;
  updateRecordingHint("Finalisation de l'enregistrement…");
  try {
    mediaRecorder.stop();
  } catch (err) {
    console.error(err);
    updateRecordingHint("Arrêt impossible : " + err.message, true);
    recordBtn.disabled = false;
  }
}

function finalizeRecording() {
  const blob = new Blob(recordingChunks, { type: mediaRecorder && mediaRecorder.mimeType ? mediaRecorder.mimeType : "audio/webm" });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `enregistrement_${timestamp}.webm`;
  const previousRecordedName = lastRecordedFile ? lastRecordedFile.name : null;
  const previousRecordedSize = lastRecordedFile ? lastRecordedFile.size : null;
  const newRecordedFile = new File([blob], filename, { type: blob.type, lastModified: Date.now() });
  lastRecordedFile = newRecordedFile;

  let dataTransfer;
  try {
    dataTransfer = new DataTransfer();
  } catch (err) {
    console.error(err);
  }

  if (!dataTransfer) {
    updateRecordingHint("Enregistrement prêt mais impossible de l'ajouter automatiquement. Téléchargez-le manuellement.", true);
    stopRecordingStreams();
    isRecording = false;
    setRecordButtonState(false);
    if (recordBtn) recordBtn.disabled = false;
    mediaRecorder = null;
    recordingChunks = [];
    return;
  }

  dataTransfer.items.add(newRecordedFile);

  Array.from(filesInput.files || []).forEach((file) => {
    if (previousRecordedName && file.name === previousRecordedName && file.size === previousRecordedSize) {
      return;
    }
    if (file.name === newRecordedFile.name && file.size === newRecordedFile.size) {
      return;
    }
    dataTransfer.items.add(file);
  });

  filesInput.files = dataTransfer.files;
  filesInput.dispatchEvent(new Event("change"));

  updateRecordingHint(`Enregistrement ajouté : ${filename}`);

  isRecording = false;
  setRecordButtonState(false);
  if (recordBtn) recordBtn.disabled = false;
  stopRecordingStreams();
  mediaRecorder = null;
  recordingChunks = [];
  stopRecordTimer();
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
  const files = Array.from(filesInput.files || []);
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
  if (!filesInput.files.length) {
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
  Array.from(filesInput.files).forEach(f => fd.append("files", f, f.name));


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
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  currentJobId = null;
  lastLogLength = 0;
  isRunning = false;
  setTranscribing(false);

  if (isRecording && mediaRecorder) {
    try { mediaRecorder.stop(); } catch (_) { /* noop */ }
  }
  stopRecordingStreams();
  mediaRecorder = null;
  recordingChunks = [];
  isRecording = false;
  lastRecordedFile = null;
  setRecordButtonState(false);
  if (recordBtn) {
    recordBtn.disabled = false;
  }
  updateRecordingHint("");
  stopRecordTimer(true);

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
filesInput.addEventListener("change", computeTotalDuration);

fillModelOptions();
fillLangOptions();
updateEstimate();

if (recordBtn) {
  setRecordButtonState(false);
  resetRecordTimerDisplay();
  recordBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecordingAction();
    } else {
      startRecording();
    }
  });
}
