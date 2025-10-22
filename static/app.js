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
const systemSourceWrap = document.getElementById("system-source-wrap");
const systemSourceSelect = document.getElementById("system-source");
const systemSourceStatus = document.getElementById("system-source-status");
const systemSourceRefreshBtn = document.getElementById("system-source-refresh");

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

const SYSTEM_AUDIO_KEYWORDS = [
  "mix",
  "stér",
  "stereo",
  "système",
  "system",
  "pc",
  "ordinateur",
  "loopback",
  "haut-parleur",
  "haut parleur",
  "speaker",
  "what u hear",
  "output",
  "internal",
  "mixage",
];

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
let lastRecordedFile = null;
let isRecording = false;
let recordTimerInterval = null;
let recordStartTime = 0;
let audioInputs = [];
let audioContext = null;
let mixedAudioNodes = [];
let currentSystemAudioLabel = "";
let hadSystemAudio = false;

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

function normalizeLevel(level) {
  if (level === true) return "error";
  if (level === false || !level) return "info";
  if (typeof level === "string") {
    const lowered = level.toLowerCase();
    if (["error", "warning", "success", "info"].includes(lowered)) {
      return lowered;
    }
  }
  return "info";
}

function updateRecordingHint(text = "", level = "info") {
  if (!recordingHint) return;
  const normalized = normalizeLevel(level);
  recordingHint.textContent = text;
  recordingHint.classList.remove("is-error", "is-warning", "is-success");
  if (normalized === "error") {
    recordingHint.classList.add("is-error");
  } else if (normalized === "warning") {
    recordingHint.classList.add("is-warning");
  } else if (normalized === "success") {
    recordingHint.classList.add("is-success");
  }
}

function setSystemSourceStatus(text = "", level = "info") {
  if (!systemSourceStatus) return;
  const normalized = normalizeLevel(level);
  systemSourceStatus.textContent = text;
  systemSourceStatus.classList.remove("is-error", "is-warning", "is-success");
  if (normalized === "error") {
    systemSourceStatus.classList.add("is-error");
  } else if (normalized === "warning") {
    systemSourceStatus.classList.add("is-warning");
  } else if (normalized === "success") {
    systemSourceStatus.classList.add("is-success");
  }
}

function ensureAudioContext() {
  if (typeof window === "undefined") return null;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  if (!audioContext || audioContext.state === "closed") {
    audioContext = new Ctx();
  }
  if (audioContext.state === "suspended") {
    audioContext.resume().catch(() => {});
  }
  return audioContext;
}

function clearMixedAudioNodes() {
  mixedAudioNodes.forEach(node => {
    try { node.disconnect(); } catch (_) { /* noop */ }
  });
  mixedAudioNodes = [];
}

function mixStreams(streams) {
  const ctx = ensureAudioContext();
  if (!ctx) return null;
  clearMixedAudioNodes();
  const destination = ctx.createMediaStreamDestination();
  streams.forEach(stream => {
    if (!stream) return;
    const tracks = stream.getAudioTracks ? stream.getAudioTracks() : [];
    if (!tracks.length) return;
    const source = ctx.createMediaStreamSource(new MediaStream(tracks));
    source.connect(destination);
    mixedAudioNodes.push(source);
  });
  const mixedStream = destination.stream;
  return mixedStream && mixedStream.getAudioTracks().length ? mixedStream : null;
}

function stopRecordingStreams() {
  recordingStreams.forEach((stream) => {
    if (!stream) return;
    stream.getTracks().forEach(track => {
      try { track.stop(); } catch (_) { /* noop */ }
    });
  });
  recordingStreams = [];
  clearMixedAudioNodes();
  if (audioContext && audioContext.state === "running") {
    audioContext.suspend().catch(() => {});
  }
  currentSystemAudioLabel = "";
  hadSystemAudio = false;
}

function deviceLabelMatchesSystem(label = "") {
  if (!label) return false;
  const lower = label.toLowerCase();
  return SYSTEM_AUDIO_KEYWORDS.some(keyword => lower.includes(keyword));
}

function detectAutoSystemDevice() {
  return audioInputs.find(device => deviceLabelMatchesSystem(device.label));
}

function buildMicConstraints() {
  return {
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  };
}

function buildSystemConstraints(deviceId) {
  const audio = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 2,
    sampleRate: 48000,
  };
  if (deviceId) {
    audio.deviceId = { exact: deviceId };
  }
  return { audio };
}

async function refreshSystemDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices || !systemSourceSelect) {
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    audioInputs = devices.filter(d => d && d.kind === "audioinput");
  } catch (err) {
    console.error(err);
    audioInputs = [];
  }

  const previousValue = systemSourceSelect.value || "auto";
  systemSourceSelect.innerHTML = "";

  const fragment = document.createDocumentFragment();

  const autoOpt = document.createElement("option");
  autoOpt.value = "auto";
  autoOpt.textContent = "Auto (mixage système si disponible)";
  fragment.appendChild(autoOpt);

  const noneOpt = document.createElement("option");
  noneOpt.value = "none";
  noneOpt.textContent = "Aucun (micro uniquement)";
  fragment.appendChild(noneOpt);

  audioInputs.forEach(device => {
    const opt = document.createElement("option");
    opt.value = device.deviceId;
    opt.textContent = device.label || `Entrée audio (${audioInputs.indexOf(device) + 1})`;
    fragment.appendChild(opt);
  });

  if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) {
    const screenOpt = document.createElement("option");
    screenOpt.value = "screen";
    screenOpt.textContent = "Capture d'écran (audio système)";
    fragment.appendChild(screenOpt);
  }

  systemSourceSelect.appendChild(fragment);

  const availableValues = Array.from(systemSourceSelect.options).map(opt => opt.value);
  if (availableValues.includes(previousValue)) {
    systemSourceSelect.value = previousValue;
  } else {
    systemSourceSelect.value = availableValues.includes("auto") ? "auto" : availableValues[0] || "none";
  }

  updateSystemSourceSummary();
}

function updateSystemSourceSummary() {
  if (!systemSourceSelect) return;
  const value = systemSourceSelect.value;
  if (value === "none") {
    setSystemSourceStatus("Micro uniquement.");
    return;
  }
  if (value === "auto") {
    const device = detectAutoSystemDevice();
    if (device) {
      setSystemSourceStatus(`Auto : ${device.label}`);
    } else if (audioInputs.length) {
      setSystemSourceStatus("Auto : aucun mixage système détecté. Sélectionnez un périphérique dans la liste.", "warning");
    } else {
      setSystemSourceStatus("Aucun périphérique audio détecté.", "warning");
    }
    return;
  }
  if (value === "screen") {
    setSystemSourceStatus("Le navigateur demandera de partager un écran avec l'audio.", "warning");
    return;
  }
  const device = audioInputs.find(d => d.deviceId === value);
  if (device) {
    setSystemSourceStatus(`Sélection : ${device.label || "Périphérique audio"}`);
  } else {
    setSystemSourceStatus("Périphérique introuvable. Actualisez la liste.", "warning");
  }
}

async function tryDirectSystemAudio(micStream) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return null;
  const micTrack = micStream && micStream.getAudioTracks ? micStream.getAudioTracks()[0] : null;
  const micLabel = micTrack ? micTrack.label : "";
  const micDeviceId = micTrack && micTrack.getSettings ? micTrack.getSettings().deviceId : null;
  const attempts = [
    { audio: { captureSystemAudio: true, channelCount: 2, sampleRate: 48000, echoCancellation: false, noiseSuppression: false, autoGainControl: false } },
    { audio: { systemAudio: "include", channelCount: 2, sampleRate: 48000, echoCancellation: false, noiseSuppression: false, autoGainControl: false } },
    { audio: { channelCount: 2, sampleRate: 48000, echoCancellation: false, noiseSuppression: false, autoGainControl: false, advanced: [{ systemAudio: "include" }] } },
  ];

  for (const constraints of attempts) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      const tracks = stream.getAudioTracks ? stream.getAudioTracks() : [];
      if (!tracks.length) {
        stream.getTracks().forEach(track => track.stop());
        continue;
      }
      const track = tracks[0];
      const label = track.label || "";
      const deviceId = track.getSettings ? track.getSettings().deviceId : null;
      if ((micLabel && label && label === micLabel) || (micDeviceId && deviceId && micDeviceId === deviceId)) {
        stream.getTracks().forEach(t => t.stop());
        continue;
      }
      if (!deviceLabelMatchesSystem(label)) {
        // Avoid mistakenly capturing a second microphone.
        stream.getTracks().forEach(t => t.stop());
        continue;
      }
      return { stream, label: label || "Mixage système" };
    } catch (err) {
      // Ignore errors and continue trying other constraints.
    }
  }
  return null;
}

async function obtainSystemAudioStreamFromSelection() {
  if (!systemSourceSelect) {
    return { stream: null, label: "", message: "", level: "info" };
  }

  const value = systemSourceSelect.value || "auto";
  if (value === "none") {
    setSystemSourceStatus("Micro uniquement.");
    return { stream: null, label: "", message: "Le son du PC est désactivé.", level: "info" };
  }

  if (!audioInputs.length) {
    await refreshSystemDevices();
  }

  if (value === "auto") {
    const device = detectAutoSystemDevice();
    if (!device) {
      const message = "Aucun périphérique de mixage système détecté. Activez \"Stereo Mix\" ou choisissez un périphérique manuel.";
      setSystemSourceStatus(message, "warning");
      return { stream: null, label: "", message, level: "warning" };
    }
    return obtainStreamForDevice(device);
  }

  if (value === "screen") {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      const message = "La capture d'écran n'est pas disponible dans ce navigateur.";
      setSystemSourceStatus(message, "error");
      return { stream: null, label: "", message, level: "error" };
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ audio: { systemAudio: "include", suppressLocalAudioPlayback: false }, video: false });
      setSystemSourceStatus("Capture d'écran avec audio système activée.", "success");
      return { stream, label: "Capture d'écran" };
    } catch (err) {
      const message = err && err.message ? err.message : "Capture d'écran annulée.";
      setSystemSourceStatus(message, "warning");
      return { stream: null, label: "", message, level: "warning" };
    }
  }

  const device = audioInputs.find(d => d.deviceId === value);
  if (!device) {
    const message = "Périphérique introuvable. Actualisez la liste.";
    setSystemSourceStatus(message, "warning");
    return { stream: null, label: "", message, level: "warning" };
  }
  return obtainStreamForDevice(device);
}

async function obtainStreamForDevice(device) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia(buildSystemConstraints(device.deviceId));
    setSystemSourceStatus(`Son du PC : ${device.label || "mixage"}.`, "success");
    return { stream, label: device.label || "Mixage système" };
  } catch (err) {
    const message = err && err.message ? err.message : "Accès refusé au périphérique audio.";
    setSystemSourceStatus(message, "error");
    return { stream: null, label: device.label || "", message, level: "error" };
  }
}

async function getSystemAudioStream(micStream) {
  const direct = await tryDirectSystemAudio(micStream);
  if (direct && direct.stream) {
    setSystemSourceStatus(`Son du PC : ${direct.label || "capture automatique"}.`, "success");
    return { stream: direct.stream, label: direct.label || "Mixage système" };
  }
  return obtainSystemAudioStreamFromSelection();
}

async function startRecording() {
  if (!recordBtn) return;
  if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
    updateRecordingHint("Enregistrement non supporté sur ce navigateur.", true);
    return;
  }

  recordBtn.disabled = true;
  updateRecordingHint("Initialisation de l'enregistrement…");

  let micStream = null;
  let systemInfo = { stream: null, label: "" };

  try {
    micStream = await navigator.mediaDevices.getUserMedia(buildMicConstraints());
    await refreshSystemDevices();
    systemInfo = await getSystemAudioStream(micStream);

    recordingStreams = [micStream];
    const streamsToMix = [micStream];
    if (systemInfo && systemInfo.stream) {
      recordingStreams.push(systemInfo.stream);
      streamsToMix.push(systemInfo.stream);
    }

    const mixedStream = mixStreams(streamsToMix);

    if (!mixedStream) {
      throw new Error("Impossible de mixer l'audio dans ce navigateur.");
    }

    recordingStreams.push(mixedStream);
    recordingChunks = [];
    mediaRecorder = new MediaRecorder(mixedStream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordingChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = finalizeRecording;
    mediaRecorder.onerror = (event) => {
      console.error(event.error || event);
      updateRecordingHint("Erreur d'enregistrement : " + (event.error ? event.error.message : event.message || event.type), "error");
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
    hadSystemAudio = !!(systemInfo && systemInfo.stream);
    currentSystemAudioLabel = systemInfo && systemInfo.label ? systemInfo.label : "";

    if (hadSystemAudio) {
      updateRecordingHint("Enregistrement en cours… micro + audio du PC.", "success");
    } else {
      const warningMessage = (systemInfo && systemInfo.message) ? systemInfo.message : "Enregistrement en cours… son du PC introuvable, micro uniquement.";
      const levelRaw = (systemInfo && systemInfo.level) ? systemInfo.level : "warning";
      const normalized = normalizeLevel(levelRaw);
      const displayLevel = normalized === "error" ? "error" : (normalized === "info" ? "info" : "warning");
      updateRecordingHint(warningMessage, displayLevel);
    }
    startRecordTimer();
  } catch (err) {
    console.error(err);
    updateRecordingHint("Impossible de démarrer : " + (err && err.message ? err.message : err), "error");
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

  const usedSystemAudio = hadSystemAudio;
  const systemLabel = currentSystemAudioLabel;
  const labelInfo = usedSystemAudio ? (systemLabel ? `micro + ${systemLabel}` : "micro + PC") : "micro uniquement";
  updateRecordingHint(`Enregistrement ajouté (${labelInfo}) : ${filename}`, usedSystemAudio ? "success" : "warning");

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
  if (systemSourceSelect) {
    updateSystemSourceSummary();
  }

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

if (!navigator.mediaDevices && systemSourceWrap) {
  systemSourceWrap.style.display = "none";
}

if (systemSourceRefreshBtn) {
  systemSourceRefreshBtn.addEventListener("click", () => {
    refreshSystemDevices();
  });
}

if (systemSourceSelect) {
  systemSourceSelect.addEventListener("change", updateSystemSourceSummary);
}

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    refreshSystemDevices();
  });
}

if (systemSourceSelect) {
  refreshSystemDevices();
}

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
