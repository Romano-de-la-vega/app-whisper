const canvas = document.getElementById("bg-waveform");
let ctx = null;
let animationId = null;
const BAR_COUNT = 60;
const SMOOTHING = 0.05;
const STEP = 0.02; // target change per frame
const MIN_LEVEL = 0.1;
const MAX_LEVEL = 0.5;
const BASE_LEVEL = (MIN_LEVEL + MAX_LEVEL) / 2;
let samples = new Array(BAR_COUNT).fill(BASE_LEVEL);
let targets = new Array(BAR_COUNT).fill(BASE_LEVEL);

function resize() {
  if (!canvas) return;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function updateTargets() {
  targets = targets.map((v) => {
    const next = v + (Math.random() * 2 - 1) * STEP;
    return Math.min(MAX_LEVEL, Math.max(MIN_LEVEL, next));
  });
}

function updateSamples() {
  updateTargets();
  samples = samples.map((v, i) => v + (targets[i] - v) * SMOOTHING);
}

function draw() {
  if (!ctx) return;
  resize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const color = getComputedStyle(document.documentElement).getPropertyValue("--accent") || "#fff";
  ctx.fillStyle = color.trim();
  ctx.globalAlpha = 0.35;
  const barWidth = canvas.width / BAR_COUNT;
  updateSamples();
  samples.forEach((v, i) => {
    const x = i * barWidth;
    const h = v * canvas.height;
    ctx.fillRect(x, canvas.height - h, barWidth * 0.7, h);
  });
  animationId = requestAnimationFrame(draw);
}

export function startWaveform() {
  if (!canvas) return;
  ctx = canvas.getContext("2d");
  if (animationId) cancelAnimationFrame(animationId);
  samples.fill(BASE_LEVEL);
  targets.fill(BASE_LEVEL);
  draw();
}

export function stopWaveform() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", resize);

