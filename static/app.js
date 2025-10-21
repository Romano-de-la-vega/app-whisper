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
const recordBtn = document.getElementById("record-btn");
const recordingNote = document.getElementById("recording-note");
const startBtn = document.getElementById("start");
const resetBtn = document.getElementById("reset");

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

let recorder = null;
let recordedChunks = [];
let mixStream = null;
let displayStream = null;
let micStream = null;
let audioContext = null;
let recordingState = "idle";
let filesFromRecording = false;
let pendingRecordedFile = null;

let particlesPromise = null;
function loadParticles() {
  if (!particlesPromise) {
    particlesPromise = import("./particles.js").then(() => window.Particles);
  }
  return particlesPromise;
}

function supportsRecording() {
  return (
    typeof MediaRecorder !== "undefined" &&
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === "function" &&
    typeof navigator.mediaDevices.getDisplayMedia === "function"
  );
}

function updateRecordingNote(text = "", emphasize = false) {
  if (!recordingNote) return;
  if (text) {
    recordingNote.hidden = false;
    recordingNote.textContent = text;
    recordingNote.classList.toggle("active", !!emphasize);
  } else {
    recordingNote.hidden = true;
    recordingNote.textContent = "";
    recordingNote.classList.remove("active");
  }
}

function setRecordingState(state) {
  recordingState = state;
  if (!recordBtn) return;
  if (state === "idle") {
    recordBtn.disabled = false;
    recordBtn.textContent = "🔴 Enregistrer";
    recordBtn.classList.remove("recording");
  } else if (state === "starting") {
    recordBtn.disabled = true;
    recordBtn.textContent = "Préparation…";
    recordBtn.classList.remove("recording");
  } else if (state === "recording") {
    recordBtn.disabled = false;
    recordBtn.textContent = "⏹️ Arrêter";
    recordBtn.classList.add("recording");
  } else if (state === "saving") {
    recordBtn.disabled = true;
    recordBtn.textContent = "Traitement…";
    recordBtn.classList.remove("recording");
  }
}

function cleanupRecordingResources() {
  if (mixStream) {
    try { mixStream.getTracks().forEach(t => t.stop()); } catch (e) { /* noop */ }
  }
  if (displayStream) {
    try { displayStream.getTracks().forEach(t => t.stop()); } catch (e) { /* noop */ }
  }
  if (micStream) {
    try { micStream.getTracks().forEach(t => t.stop()); } catch (e) { /* noop */ }
  }
  mixStream = null;
  displayStream = null;
  micStream = null;
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  recorder = null;
}

async function startRecording() {
  if (!supportsRecording()) {
    alert("L'enregistrement n'est pas supporté par ce navigateur.");
    return;
  }

  try {
    pendingRecordedFile = null;
    setRecordingState("starting");
    updateRecordingNote("Sélectionne l'écran à capturer pour récupérer le son du PC…");

    displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
    displayStream.getVideoTracks().forEach(track => { track.enabled = false; });

    const handleExternalStop = () => {
      if (recordingState === "recording") {
        stopRecording();
      }
    };
    displayStream.getTracks().forEach(track => {
      track.addEventListener("ended", handleExternalStop, { once: false });
    });

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const destination = audioContext.createMediaStreamDestination();
    let connected = 0;

    const tryConnect = (stream) => {
      if (!stream) return;
      const audioTracks = stream.getAudioTracks();
      if (!audioTracks || !audioTracks.length) return;
      const onlyAudio = new MediaStream(audioTracks);
      const source = audioContext.createMediaStreamSource(onlyAudio);
      source.connect(destination);
      connected += 1;
    };

    tryConnect(displayStream);
    tryConnect(micStream);

    if (!connected) {
      throw new Error("Aucun flux audio capturé");
    }

    const preferredMime = "audio/webm;codecs=opus";
    const options = MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(preferredMime)
      ? { mimeType: preferredMime }
      : undefined;

    mixStream = destination.stream;
    recorder = options ? new MediaRecorder(mixStream, options) : new MediaRecorder(mixStream);
    recordedChunks = [];

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    recorder.onstop = async () => {
      const mimeType = recorder && recorder.mimeType ? recorder.mimeType : "audio/webm";
      const blob = new Blob(recordedChunks, { type: mimeType });
      cleanupRecordingResources();

      if (!blob.size) {
        setRecordingState("idle");
        updateRecordingNote("Enregistrement vide.");
        return;
      }

      const filename = `enregistrement_${new Date().toISOString().replace(/[\.:]/g, "-")}.webm`;
      const file = new File([blob], filename, { type: blob.type || "audio/webm" });
      pendingRecordedFile = file;

      let dt = null;
      try {
        dt = new DataTransfer();
      } catch (e) {
        if (typeof ClipboardEvent !== "undefined") {
          try { dt = new ClipboardEvent("copy").clipboardData || null; } catch (_) { dt = null; }
        }
      }

      if (dt && dt.items) {
        dt.items.add(file);
        filesInput.files = dt.files;
        filesFromRecording = true;
        filesInput.dispatchEvent(new Event("change"));
      } else {
        filesFromRecording = true;
        updateEstimateWithRecordedFile(file).catch(() => {});
      }
      updateRecordingNote(`Prêt : ${filename}`);
      setRecordingState("idle");
    };

    setRecordingState("recording");
    updateRecordingNote("Enregistrement en cours…", true);
    recorder.start();
  } catch (err) {
    console.error(err);
    cleanupRecordingResources();
    setRecordingState("idle");
    const message = err && err.message ? err.message : "Impossible de démarrer l'enregistrement.";
    updateRecordingNote(message);
  }
}

function stopRecording() {
  if (!recorder) return;
  setRecordingState("saving");
  updateRecordingNote("Finalisation de l'enregistrement…");
  try {
    recorder.stop();
  } catch (e) {
    console.error(e);
    cleanupRecordingResources();
    setRecordingState("idle");
    updateRecordingNote("Échec de l'arrêt de l'enregistrement.");
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

function handleFilesChanged() {
  computeTotalDuration();
  if (!filesFromRecording) {
    updateRecordingNote("");
    pendingRecordedFile = null;
  }
  filesFromRecording = false;
}

async function updateEstimateWithRecordedFile(file) {
  if (!file) return;
  const duration = await getAudioDuration(file);
  totalDurationMin = (duration || 0) / 60;
  updateEstimate();
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
  if (!filesInput.files.length && !pendingRecordedFile) {
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
  const filesToSend = Array.from(filesInput.files);
  if (!filesToSend.length && pendingRecordedFile) {
    filesToSend.push(pendingRecordedFile);
  }
  filesToSend.forEach(f => fd.append("files", f, f.name));


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

  if (recordingState !== "idle") {
    cleanupRecordingResources();
    setRecordingState("idle");
  }
  updateRecordingNote("");
  pendingRecordedFile = null;
  filesFromRecording = false;

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

if (recordBtn) {
  if (!supportsRecording()) {
    recordBtn.disabled = true;
    recordBtn.title = "Enregistrement non supporté sur ce navigateur";
  } else {
    recordBtn.addEventListener("click", () => {
      if (recordingState === "recording") {
        stopRecording();
      } else if (recordingState === "idle") {
        startRecording();
      }
    });
  }
}

// ====== Init ======
modeSelect.addEventListener("change", () => { fillModelOptions(); updateEstimate(); updateSummaryBtnLabel(); });
modelSelect.addEventListener("change", updateEstimate);
filesInput.addEventListener("change", handleFilesChanged);

fillModelOptions();
fillLangOptions();
updateEstimate();